import os
import sys
import math
import tqdm
import time
import torch
import torch.nn as nn
import random
import datetime
import numpy as np
import torch.distributed as dist

from pathlib import Path
from torchvision import transforms
from collections import defaultdict, deque
from PIL import ImageFilter, ImageOps
from typing import Optional

import source.vision_transformer as vits
from source.utils import update_state_dict
from eval_knn import extract_multiple_features, knn_classifier


def hydra_argv_remapper(argv_map):
    """
    Call this function before main
    argv_map is a dict that remaps specific args to something else that hydra will gracefully not choke on
        ex: {'--foo':'standard.hydra.override.foo', '--bar':'example.bar'}
    workaround hydra behaviour with command line flags
    kindly given at: https://github.com/facebookresearch/hydra/issues/446#issuecomment-881031746
    """

    argv = sys.argv

    # Remap the args
    for k in argv_map.keys():
        if k in argv:
            i = argv.index(k)
            new_arg = f"{argv_map[k]}={argv[i].split('=')[-1]}"
            argv.append(new_arg)
            del argv[i]

    # Replace sys.argv with our remapped argv
    sys.argv = argv


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PatchDataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        global_crop_size = 224
        local_crop_size = 96

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crop_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, x):
        crops = []
        crops.append(self.global_transfo1(x))
        crops.append(self.global_transfo2(x))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(x))
        return crops


class RegionDataAugmentationDINO(object):
    """
    Modified Data Augmentaton for DINO for [region_size x region_size] resolutions for performing local / global crops on features in image grid
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_number,
        local_crops_scale,
        region_size: int = 4096,
        patch_size: int = 256,
    ):
        self.npatch = int(region_size // patch_size)
        global_crop_size = int(global_crops_scale * self.npatch)
        local_crop_size = int(local_crops_scale * self.npatch)

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomCrop(global_crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomCrop(global_crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomCrop(local_crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def __call__(self, x):
        crops = []
        x = x.unfold(0, self.npatch, self.npatch).transpose(
            0, 1
        )  # [m, 384] -> [npatch, 384, npatch] -> [384, npatch, npatch]
        crops.append(self.global_transfo1(x))
        crops.append(self.global_transfo2(x))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(x))
        return crops


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, gpu_id):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        if gpu_id == -1:
            main_device = "cuda"
        else:
            main_device = torch.device(f"cuda:{gpu_id}")
        t = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device=main_device
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, gpu_id):
        for meter in self.meters.values():
            meter.synchronize_between_processes(gpu_id)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    tqdm.tqdm.write(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    tqdm.tqdm.write(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        tqdm.tqdm.write(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


class EarlyStoppingDINO:
    """
    Leverage a downstream classification task to know if teacher still outperforms student
    """

    def __init__(
        self,
        tracking: str,
        min_max: str,
        patience: int = 20,
        min_epoch: int = 50,
        checkpoint_dir: Optional[Path] = None,
        save_every: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.tracking = tracking
        self.min_max = min_max
        self.patience = patience
        self.min_epoch = min_epoch
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.verbose = verbose

        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, results, snapshot):
        if results is not None:
            teacher_score = results["teacher"][self.tracking]
            student_score = results["student"][self.tracking]

            if self.min_max == "min":
                teacher_score = -1 * teacher_score
                student_score = -1 * student_score

            if self.best_score is None or (
                teacher_score >= self.best_score and teacher_score > student_score
            ):
                self.best_score = teacher_score
                torch.save(snapshot, Path(self.checkpoint_dir, "best.pt"))
                self.counter = 0

            elif teacher_score < self.best_score or teacher_score <= student_score:
                self.counter += 1
                if epoch <= self.min_epoch + 1 and self.verbose:
                    tqdm.tqdm.write(
                        f"EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}"
                    )
                elif self.verbose:
                    tqdm.tqdm.write(
                        f"EarlyStopping counter: {self.counter}/{self.patience}"
                    )
                if self.counter >= self.patience and epoch > self.min_epoch:
                    self.early_stop = True

        if self.save_every and epoch % self.save_every == 0:
            fname = f"snapshot_epoch_{epoch:03}.pt"
            torch.save(snapshot, Path(self.checkpoint_dir, fname))

        # override latest
        torch.save(snapshot, Path(self.checkpoint_dir, "latest.pt"))


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def start_from_checkpoint(ckpt_path, model):
    """
    Re-start from checkpoint
    """
    if not Path(ckpt_path).is_file():
        return
    print(f"Pretrained weights found at {ckpt_path}")

    # open checkpoint file
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["teacher"]
    state_dict, msg = update_state_dict(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=False)
    print(msg)


def resume_from_checkpoint(ckpt_path, verbose: bool = True, **kwargs):
    """
    Re-start from checkpoint
    """
    if not Path(ckpt_path).is_file():
        return
    if verbose:
        print(f"Found checkpoint at {ckpt_path}")

    # open checkpoint file
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                sd = checkpoint[key]
                nn.modules.utils.consume_prefix_in_state_dict_if_present(sd, "module.")
                msg = value.load_state_dict(sd, strict=False)
                if verbose:
                    print(
                        f"=> loaded '{key}' from checkpoint: '{ckpt_path}' with msg {msg}"
                    )
            except TypeError:
                try:
                    sd = checkpoint[key]
                    nn.modules.utils.consume_prefix_in_state_dict_if_present(
                        sd, "module."
                    )
                    msg = value.load_state_dict(sd)
                    if verbose:
                        print(f"=> loaded '{key}' from checkpoint: '{ckpt_path}'")
                except ValueError:
                    if verbose:
                        print(
                            f"=> failed to load '{key}' from checkpoint: '{ckpt_path}'"
                        )
        elif verbose:
            print(f"=> key '{key}' not found in checkpoint: '{ckpt_path}'")
    return epoch


def cosine_scheduler(
    base_value,
    final_value,
    nepochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(nepochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == nepochs * niter_per_ep
    return schedule


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(cfg):
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.gpu = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
        cfg.world_size = int(os.environ["SLURM_NTASKS"])
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        cfg.rank, cfg.gpu, cfg.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=cfg.dist_url,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )

    torch.cuda.device(cfg.gpu)
    dist.barrier()
    setup_for_distributed(cfg.rank == 0)


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    nepochs,
    fp16_scaler,
    clip_grad,
    freeze_last_layer,
    gpu_id,
):
    metric_logger = MetricLogger(delimiter="  ")
    with tqdm.tqdm(
        data_loader,
        desc=(f"Epoch [{epoch+1}/{nepochs}]"),
        unit=" img",
        ncols=80,
        unit_scale=data_loader.batch_size,
        leave=False,
        file=sys.stdout,
        disable=not (gpu_id in [-1, 0]),
    ) as t:
        for it, (images, _) in enumerate(t):
            # update weight decay and learning rate according to their schedule
            it = len(data_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            # move images to gpu
            if gpu_id == -1:
                images = [im.cuda(non_blocking=True) for im in images]
            else:
                device = torch.device(f"cuda:{gpu_id}")
                images = [im.to(device, non_blocking=True) for im in images]
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(
                    images[:2]
                )  # only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                tqdm.tqdm.write(
                    "Loss is {}, stopping training".format(loss.item()), force=True
                )
                sys.exit(1)

            # student update
            optimizer.zero_grad()
            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                if clip_grad:
                    param_norms = clip_gradients(student, clip_grad)
                cancel_gradients_last_layer(epoch, student, freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if clip_grad:
                    fp16_scaler.unscale_(
                        optimizer
                    )  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = clip_gradients(student, clip_grad)
                cancel_gradients_last_layer(epoch, student, freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                if torch.cuda.device_count() > 1:
                    student_params = student.module.parameters()
                else:
                    student_params = student.parameters()
                for param_q, param_k in zip(
                    student_params, teacher_without_ddp.parameters()
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(gpu_id)
    # print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return train_stats


def load_weights(model, state_dict):
    # remove `module.` prefix induced by DDP
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    # remove `backbone.` prefix induced by multicrop wrapper
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "backbone.")
    # state_dict, msg = update_state_dict(model.state_dict(), state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    if len(msg.missing_keys) > 0:
        tqdm.tqdm.write(str(msg))
    else:
        tqdm.tqdm.write("All keys matched successfully")


def tune_one_epoch(
    epoch,
    student: nn.Module,
    teacher: nn.Module,
    train_dataloader,
    test_dataloader,
    features_dir: Path,
    arch: str,
    patch_size: int,
    drop_path_rate: float,
    k: int,
    temperature: float,
    distributed: bool,
    save_features: bool = False,
    use_cuda: bool = False,
):
    student_model = vits.__dict__[arch](
        patch_size=patch_size, drop_path_rate=drop_path_rate, num_classes=0
    )
    teacher_model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    tqdm.tqdm.write(f"Teacher & student models {arch} {patch_size}x{patch_size} built.")
    student_model.cuda()
    teacher_model.cuda()
    tqdm.tqdm.write(f"Loading epoch {epoch} weights...")
    student_weights = student.state_dict()
    teacher_weights = teacher.state_dict()
    load_weights(student_model, student_weights)
    load_weights(teacher_model, teacher_weights)
    student_model.eval()
    teacher_model.eval()

    # ============ extract student features ============
    tqdm.tqdm.write("Extracting features for train set...")
    train_features, train_labels = extract_multiple_features(
        student_model, teacher_model, train_dataloader, distributed, use_cuda
    )
    tqdm.tqdm.write("Extracting features for test set...")
    test_features, test_labels = extract_multiple_features(
        student_model, teacher_model, test_dataloader, distributed, use_cuda
    )

    teacher_train_features, teacher_test_features = (
        train_features["teacher"],
        test_features["teacher"],
    )
    student_train_features, student_test_features = (
        train_features["student"],
        test_features["student"],
    )

    # save features and labels
    if save_features and is_main_process():
        for name, feats in train_features.items():
            torch.save(feats.cpu(), Path(features_dir, f"{name}_train_feat.pth"))
        for name, feats in train_features.items():
            torch.save(feats.cpu(), Path(features_dir, f"{name}_test_feat.pth"))
        torch.save(train_labels.cpu(), Path(features_dir, "train_labels.pth"))
        torch.save(test_labels.cpu(), Path(features_dir, "test_labels.pth"))

    results = defaultdict(dict)
    if is_main_process():
        assert len(torch.unique(train_labels)) == len(
            torch.unique(test_labels)
        ), "train & test dataset have different number of classes!"
        num_classes = len(torch.unique(train_labels))
        if use_cuda:
            teacher_train_features, teacher_test_features = (
                teacher_train_features.cuda(),
                teacher_test_features.cuda(),
            )
            student_train_features, student_test_features = (
                student_train_features.cuda(),
                student_test_features.cuda(),
            )
            train_labels, test_labels = train_labels.cuda(), test_labels.cuda()

        tqdm.tqdm.write("Features are ready!\nStarting kNN classification.")
        teacher_acc, teacher_auc = knn_classifier(
            teacher_train_features,
            train_labels,
            teacher_test_features,
            test_labels,
            k,
            temperature,
            num_classes,
        )
        student_acc, student_auc = knn_classifier(
            student_train_features,
            train_labels,
            student_test_features,
            test_labels,
            k,
            temperature,
            num_classes,
        )
        results["teacher"].update({"acc": teacher_acc, "auc": teacher_auc})
        results["student"].update({"acc": student_acc, "auc": student_auc})

    return results
