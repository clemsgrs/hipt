import os
import sys
import tqdm
import time
import json
import hydra
import wandb
import shutil
import random
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd

from pathlib import Path
from omegaconf import DictConfig
from torchvision import datasets

import source.vision_transformer as vits

from source.utils import initialize_wandb, compute_time, update_log_dict
from source.components import DINOLoss
from eval_knn import prepare_data
from utils import (
    PatchDataAugmentationDINO,
    MultiCropWrapper,
    EarlyStoppingDINO,
    train_one_epoch,
    tune_one_epoch,
    fix_random_seeds,
    has_batchnorms,
    get_params_groups,
    resume_from_checkpoint,
    cosine_scheduler,
    get_world_size,
    is_main_process,
)


@hydra.main(
    version_base="1.2.0", config_path="../config/pretraining", config_name="patch"
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
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        # set up wandb
        if cfg.wandb.enable:
            key = os.environ.get("WANDB_API_KEY")
            wandb_run = initialize_wandb(cfg, key=key)
            wandb_run.define_metric("epoch", summary="max")
            run_id = wandb_run.id
    else:
        run_id = ""

    if distributed:
        obj = [run_id]
        torch.distributed.broadcast_object_list(
            obj, 0, device=torch.device(f"cuda:{gpu_id}")
        )
        run_id = obj[0]

    fix_random_seeds(cfg.seed)
    cudnn.benchmark = True

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    snapshot_dir = Path(output_dir, "snapshots")
    features_dir = Path(output_dir, "features")
    if not cfg.resume and is_main_process():
        if output_dir.exists():
            print(f"WARNING: {output_dir} already exists! Deleting its content...")
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        if cfg.early_stopping.tune_every and cfg.early_stopping.knn.save_features:
            features_dir.mkdir(exist_ok=True, parents=True)

    # preparing data
    if is_main_process():
        print(f"Loading data...")

    # ============ preparing tuning data ============
    if is_main_process() and cfg.early_stopping.tune_every:
        # only do it from master rank as tuning is not being run distributed for now
        train_df = pd.read_csv(cfg.early_stopping.downstream.train_csv)
        test_df = pd.read_csv(cfg.early_stopping.downstream.test_csv)
        downstream_train_loader, downstream_test_loader = prepare_data(
            train_df,
            test_df,
            cfg.early_stopping.downstream.batch_size_per_gpu,
            False,
            cfg.early_stopping.downstream.num_workers,
            cfg.early_stopping.downstream.label_name,
        )
        print(
            f"Tuning data loaded with {len(downstream_train_loader.dataset)} train patches and {len(downstream_test_loader.dataset)} test patches."
        )

    transform = PatchDataAugmentationDINO(
        cfg.aug.global_crops_scale,
        cfg.aug.local_crops_scale,
        cfg.aug.local_crops_number,
    )

    # ============ preparing training data ============
    dataset_loading_start_time = time.time()
    dataset = datasets.ImageFolder(cfg.data_dir, transform=transform)
    dataset_loading_end_time = time.time() - dataset_loading_start_time
    total_time_str = str(datetime.timedelta(seconds=int(dataset_loading_end_time)))
    if is_main_process():
        print(f"Pretraining data loaded in {total_time_str} ({len(dataset)} patches)")

    if cfg.training.pct:
        print(f"Pre-training on {cfg.training.pct*100}% of the data")
        nsample = int(cfg.training.pct * len(dataset))
        idxs = random.sample(range(len(dataset)), k=nsample)
        dataset = torch.utils.data.Subset(dataset, idxs)

    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.training.batch_size_per_gpu,
        num_workers=cfg.speed.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # building student and teacher networks
    if is_main_process():
        print(f"Building student and teacher networks...")
    student = vits.__dict__[cfg.model.arch](
        patch_size=cfg.model.patch_size,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    teacher = vits.__dict__[cfg.model.arch](patch_size=cfg.model.patch_size)
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(
        student,
        vits.DINOHead(
            embed_dim,
            cfg.model.out_dim,
            use_bn=cfg.model.use_bn_in_head,
            norm_last_layer=cfg.model.norm_last_layer,
        ),
    )
    teacher = MultiCropWrapper(
        teacher,
        vits.DINOHead(
            embed_dim,
            cfg.model.out_dim,
            use_bn=cfg.model.use_bn_in_head,
        ),
    )

    # move networks to gpu
    if distributed:
        student, teacher = student.to(gpu_id), teacher.to(gpu_id)
    else:
        student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if has_batchnorms(student) and distributed:
        # we need DDP wrapper to have synchro batch norms working...
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[gpu_id], output_device=gpu_id
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    if distributed:
        student = nn.parallel.DistributedDataParallel(
            student, device_ids=[gpu_id], output_device=gpu_id
        )

    # teacher and student start with the same weights
    student_sd = student.state_dict()
    nn.modules.utils.consume_prefix_in_state_dict_if_present(student_sd, "module.")
    teacher_without_ddp.load_state_dict(student_sd)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # total number of crops = 2 global crops + local_crops_number
    crops_number = cfg.aug.local_crops_number + 2
    dino_loss = DINOLoss(
        cfg.model.out_dim,
        crops_number,
        cfg.model.warmup_teacher_temp,
        cfg.model.teacher_temp,
        cfg.model.warmup_teacher_temp_epochs,
        cfg.training.nepochs,
    )
    if distributed:
        dino_loss = dino_loss.to(gpu_id)
    else:
        dino_loss = dino_loss.cuda()

    params_groups = get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)

    # for mixed precision training
    fp16_scaler = None
    if cfg.speed.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    assert (
        cfg.training.nepochs >= cfg.training.warmup_epochs
    ), f"nepochs ({cfg.training.nepochs}) must be greater than or equal to warmup_epochs ({cfg.training.warmup_epochs})"
    base_lr = (
        cfg.optim.lr * (cfg.training.batch_size_per_gpu * get_world_size()) / 256.0
    )
    lr_schedule = cosine_scheduler(
        base_lr,
        cfg.optim.lr_scheduler.min_lr,
        cfg.training.nepochs,
        len(data_loader),
        warmup_epochs=cfg.training.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        cfg.optim.lr_scheduler.weight_decay,
        cfg.optim.lr_scheduler.weight_decay_end,
        cfg.training.nepochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        cfg.model.momentum_teacher, 1, cfg.training.nepochs, len(data_loader)
    )
    if is_main_process():
        print(f"Models built, kicking off training")

    epochs_run = 0

    # leverage torch native fault tolerance
    snapshot_path = Path(snapshot_dir, "latest.pt")
    if distributed:
        if snapshot_path.exists():
            if is_main_process():
                print("Loading snapshot")
            loc = f"cuda:{gpu_id}"
            snapshot = torch.load(snapshot_path, map_location=loc)
            epochs_run = snapshot["epoch"]
            student.load_state_dict(snapshot["student"])
            teacher.load_state_dict(snapshot["teacher"])
            optimizer.load_state_dict(snapshot["optimizer"])
            dino_loss.load_state_dict(snapshot["dino_loss"])
            if fp16_scaler is not None:
                fp16_scaler.load_state_dict(snapshot["fp16_scaler"])
            if is_main_process():
                print(f"Resuming training from snapshot at epoch {epochs_run}")

    elif cfg.resume:
        ckpt_path = Path(cfg.resume_from_checkpoint)
        epochs_run = resume_from_checkpoint(
            ckpt_path,
            verbose=(gpu_id in [-1, 0]),
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
        if is_main_process():
            print(f"Resuming training from checkpoint at epoch {epochs_run}")

    early_stopping = EarlyStoppingDINO(
        cfg.early_stopping.tracking,
        cfg.early_stopping.min_max,
        cfg.early_stopping.patience,
        cfg.early_stopping.min_epoch,
        checkpoint_dir=snapshot_dir,
        save_every=cfg.early_stopping.save_every,
        verbose=True,
    )

    stop = False
    start_time = time.time()

    with tqdm.tqdm(
        range(epochs_run, cfg.training.nepochs),
        desc=(f"DINO Pretraining"),
        unit=" epoch",
        ncols=100,
        leave=True,
        initial=epochs_run,
        total=cfg.training.nepochs,
        file=sys.stdout,
        position=0,
        disable=not is_main_process(),
    ) as t:
        for epoch in t:
            epoch_start_time = time.time()
            if cfg.wandb.enable and is_main_process():
                log_dict = {"epoch": epoch}

            if distributed:
                data_loader.sampler.set_epoch(epoch)

            # training one epoch of DINO
            train_stats = train_one_epoch(
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
                cfg.training.nepochs,
                fp16_scaler,
                cfg.training.clip_grad,
                cfg.training.freeze_last_layer,
                gpu_id,
            )

            if cfg.wandb.enable and is_main_process():
                update_log_dict("train", train_stats, log_dict, step="epoch")

            if is_main_process():
                snapshot = {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "dino_loss": dino_loss.state_dict(),
                }
                if fp16_scaler is not None:
                    snapshot["fp16_scaler"] = fp16_scaler.state_dict()

            # only run tuning on rank 0, otherwise one has to take care of gathering knn metrics from multiple gpus
            tune_results = None
            if (
                cfg.early_stopping.tune_every
                and epoch % cfg.early_stopping.tune_every == 0
                and is_main_process()
            ):
                tune_results = tune_one_epoch(
                    epoch + 1,
                    student,
                    teacher_without_ddp,
                    downstream_train_loader,
                    downstream_test_loader,
                    features_dir,
                    cfg.model.arch,
                    cfg.model.patch_size,
                    cfg.model.drop_path_rate,
                    cfg.early_stopping.knn.k,
                    cfg.early_stopping.knn.temperature,
                    False,
                    cfg.early_stopping.knn.save_features,
                    cfg.early_stopping.knn.use_cuda,
                )

                if cfg.wandb.enable and is_main_process():
                    update_log_dict("tune", tune_results, log_dict, step="epoch")

            if is_main_process():
                early_stopping(epoch, tune_results, snapshot)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            if stop:
                tqdm.tqdm.write(
                    f"Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago"
                )
                break

            # save snapshot and log to wandb
            if is_main_process():
                save_path = Path(snapshot_dir, f"epoch_{epoch:03}.pt")
                if (
                    cfg.early_stopping.save_every
                    and epoch % cfg.early_stopping.save_every == 0
                    and not save_path.is_file()
                ):
                    torch.save(snapshot, save_path)

                if cfg.wandb.enable:
                    wandb.log(log_dict, step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            if is_main_process():
                with open(Path(output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            if is_main_process():
                tqdm.tqdm.write(
                    f"End of epoch {epoch+1}/{cfg.training.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
                )

            # ensure other gpus wait until gpu_0 is finished with tuning before starting next training iteration
            if distributed:
                torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Pretraining time {}".format(total_time_str))

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # python3 -m torch.distributed.run --standalone --nproc_per_node=gpu pretrain/dino_patch.py --config-name 'debug'

    # ISSUE WITH TORCHRUN ON SOL2: USES PYTHON3.8 INSTEAD OF PYTHON3.9 FOR SOME REASON
    # torchrun --standalone pretrain/dino_patch.py --nproc_per_node=gpu --config-name 'debug'

    # m = {}
    # for i in range(torch.cuda.device_count()):
    #     m_i = {f"--local_rank={i}": "local_rank"}
    #     m.update(m_i)
    # hydra_argv_remapper(m)

    main()
