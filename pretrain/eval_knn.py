import os
import tqdm
import hydra
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from pathlib import Path
from sklearn import metrics
from omegaconf import DictConfig
from torchvision import datasets
from torchvision import transforms

import source.vision_transformer as vits
from utils import is_main_process


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if Path(pretrained_weights).is_file():
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("WARNING: provided weights don't exist, random weights will be used.")


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v


def extract_feature_pipeline(
        data_dir: str,
        features_dir: str,
        arch: str,
        patch_size: int,
        pretrained_weights: str,
        checkpoint_key: str,
        batch_size_per_gpu: int,
        distributed: bool,
        save_features: bool = False,
        use_cuda: bool = True,
        num_workers: int = 10,
    ):

    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(Path(data_dir, "train"), transform=transform)
    dataset_test = ReturnIndexDataset(Path(data_dir, "test"), transform=transform)
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    else:
        sampler = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_test)} eval imgs.")

    # ============ building network ... ============
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    print(f"Model {arch} {patch_size}x{patch_size} built.")
    model.cuda()
    print(f"Loading pretrained weights...")
    load_pretrained_weights(model, pretrained_weights, checkpoint_key)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, distributed, use_cuda)
    print("Extracting features for test set...")
    test_features = extract_features(model, data_loader_test, distributed, use_cuda)

    if is_main_process():
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_test.samples]).long()
    # save features and labels
    if save_features and is_main_process():
        torch.save(train_features.cpu(), Path(features_dir, "train_feat.pth"))
        torch.save(test_features.cpu(), Path(features_dir, "test_feat.pth"))
        torch.save(train_labels.cpu(), Path(features_dir, "train_labels.pth"))
        torch.save(test_labels.cpu(), Path(features_dir, "test_labels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, loader, distributed, use_cuda=True, multiscale=False):

    features = None

    with tqdm.tqdm(
        loader,
        desc=(f"Feature extraction"),
        unit=" slide",
        ncols=80,
        unit_scale=loader.batch_size,
        leave=True,
    ) as t:

        for i, batch in enumerate(t):

            samples, index = batch
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            if multiscale:
                feats = multi_scale(samples, model)
            else:
                feats = model(samples).clone()

            # init storage feature matrix
            if is_main_process() and features is None:
                features = torch.zeros(len(loader.dataset), feats.shape[-1])
                if use_cuda:
                    features = features.cuda(non_blocking=True)
                t.display(f"Storing features into tensor of shape {features.shape}", pos=1)
                print()

            if distributed:
                ngpu = dist.get_world_size()
                y_all = torch.empty(ngpu, index.size(0), dtype=index.dtype, device=index.device)
                y_l = list(y_all.unbind(0))
                y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
                y_all_reduce.wait()
                index_all = torch.cat(y_l)

                # share features between processes
                feats_all = torch.empty(
                    ngpu,
                    feats.size(0),
                    feats.size(1),
                    dtype=feats.dtype,
                    device=feats.device,
                )
                output_l = list(feats_all.unbind(0))
                output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
                output_all_reduce.wait()

                # update storage feature matrix
                if is_main_process():
                    if use_cuda:
                        features.index_copy_(0, index_all, torch.cat(output_l))
                    else:
                        features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())

    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=2):
    acc, total = 0.0, 0
    test_probs = np.empty((0, num_classes))
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], min(test_labels.shape[0], 100)
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        p = F.softmax(probs, dim=1).cpu().detach().numpy()
        test_probs = np.append(test_probs, p, axis=0)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        acc = acc + correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)

    acc = acc * 100.0 / total
    if num_classes == 2:
        auc = metrics.roc_auc_score(test_labels.cpu(), test_probs[:, 1])
    else:
        print("WARNING: multi-class AUC not implemented")
        auc = -1

    return acc, auc


@hydra.main(
    version_base="1.2.0",
    config_path="../config/pretraining",
    config_name="knn",
)
def main(cfg: DictConfig):

    distributed = torch.cuda.device_count() > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        gpu_id = int(os.environ["LOCAL_RANK"])
        if gpu_id == 0:
            print(f"Distributed session successfully initialized")
    else:
        gpu_id = -1
    if is_main_process():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    cudnn.benchmark = True

    if cfg.load_features:
        train_features = torch.load(Path(cfg.features_dir, "train_feat.pth"))
        test_features = torch.load(Path(cfg.features_dir, "test_feat.pth"))
        train_labels = torch.load(Path(cfg.features_dir, "train_labels.pth"))
        test_labels = torch.load(Path(cfg.features_dir, "test_labels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(
            cfg.data_dir,
            cfg.features_dir,
            cfg.model.arch,
            cfg.model.patch_size,
            cfg.model.pretrained_weights,
            cfg.model.checkpoint_key,
            cfg.batch_size_per_gpu,
            distributed,
            save_features=cfg.save_features,
            use_cuda=cfg.speed.use_cuda,
            num_workers=cfg.speed.num_workers,
        )

    if is_main_process():
        if cfg.speed.use_cuda:
            train_features, train_labels = train_features.cuda(), train_labels.cuda()
            test_features, test_labels = test_features.cuda(), test_labels.cuda()

        print("Features are ready!\nStarting kNN classification.")
        for k in cfg.nb_knn:
            acc, auc = knn_classifier(train_features, train_labels, test_features, test_labels, k, cfg.temperature)
            print(f"{k}-NN classifier result:")
            print(f"- auc: {auc}")
            print(f"- accuracy: {acc:.2f}%")

    if distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':

    main()