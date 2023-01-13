import os
import datetime
import time
import json
import hydra
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import DictConfig
from torchvision import datasets

import vision_transformer as vits

from .components import DINOLoss
from .utils import (
    DataAugmentationDINO,
    MultiCropWrapper,
    train_one_epoch,
    init_distributed_mode,
    fix_random_seeds,
    has_batchnorms,
    get_params_groups,
    cosine_scheduler,
    get_world_size,
    restart_from_checkpoint,
    save_on_master,
    is_main_process,
)


@hydra.main(version_base="1.2.0", config_path="config", config_name="dino")
def main(cfg: DictConfig):

    init_distributed_mode(cfg)
    fix_random_seeds(cfg.seed)

    cudnn.benchmark = True

    # preparing data
    transform = DataAugmentationDINO(
        cfg.global_crops_scale,
        cfg.local_crops_scale,
        cfg.local_crops_number,
    )
    dataset = datasets.ImageFolder(cfg.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # building student and teacher networks
    student = vits.__dict__[cfg.arch](
        patch_size=cfg.patch_size,
        drop_path_rate=cfg.drop_path_rate,  # stochastic depth
    )
    teacher = vits.__dict__[cfg.arch](patch_size=cfg.patch_size)
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(
        student,
        vits.DINOHead(
            embed_dim,
            cfg.out_dim,
            use_bn=cfg.use_bn_in_head,
            norm_last_layer=cfg.norm_last_layer,
        ),
    )
    teacher = MultiCropWrapper(
        teacher,
        vits.DINOHead(embed_dim, cfg.out_dim, cfg.use_bn_in_head),
    )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[cfg.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[cfg.gpu])

    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    crops_number = (
        cfg.local_crops_number + 2
    )  # total number of crops = 2 global crops + local_crops_number
    dino_loss = DINOLoss(
        cfg.out_dim,
        crops_number,
        cfg.warmup_teacher_temp,
        cfg.teacher_temp,
        cfg.warmup_teacher_temp_epochs,
        cfg.epochs,
    ).cuda()

    params_groups = get_params_groups(student)
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler

    # for mixed precision training
    fp16_scaler = None
    if cfg.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    lr_schedule = cosine_scheduler(
        cfg.lr
        * (cfg.batch_size_per_gpu * get_world_size())
        / 256.0,  # linear scaling rule
        cfg.min_lr,
        cfg.epochs,
        len(data_loader),
        warmup_epochs=cfg.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        cfg.weight_decay,
        cfg.weight_decay_end,
        cfg.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        cfg.momentum_teacher, 1, cfg.epochs, len(data_loader)
    )

    # optionally resume training
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(cfg.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )

    start_epoch = to_restore["epoch"]
    start_time = time.time()
    for epoch in range(start_epoch, cfg.epochs):
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
            cfg.nepochs,
            fp16_scaler,
            cfg.clip_grad,
            cfg.freeze_last_layer,
        )

        # writing logs
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "dino_loss": dino_loss.state_dict(),
        }

        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        save_on_master(save_dict, os.path.join(cfg.output_dir, "checkpoint.pth"))

        if cfg.saveckp_freq and epoch % cfg.saveckp_freq == 0:
            save_on_master(
                save_dict, os.path.join(cfg.output_dir, f"checkpoint{epoch:04}.pth")
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if is_main_process():
            with (Path(cfg.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    main()
