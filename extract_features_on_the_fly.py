import os
import tqdm
import wandb
import torch
import hydra
import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from omegaconf import DictConfig

from source.dataset import RegionCoordinatesDataset, PatchDataset
from source.models import GlobalFeatureExtractor, LocalFeatureExtractor
from source.utils import (
    initialize_wandb,
    collate_coordinates,
    is_main_process,
    build_slide_level_feature,
)


@hydra.main(
    version_base="1.2.0", config_path="config/feature_extraction", config_name="default"
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
            wandb_run.define_metric("processed", summary="max")
            run_id = wandb_run.id
    else:
        run_id = ""

    if distributed:
        obj = [run_id]
        torch.distributed.broadcast_object_list(
            obj, 0, device=torch.device(f"cuda:{gpu_id}")
        )
        run_id = obj[0]

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    slide_features_dir = Path(output_dir, "slide_features")
    if cfg.save_region_features:
        region_features_dir = Path(output_dir, "region_features")
    else:
        region_features_dir = Path("/tmp/region_features")

    if not cfg.resume and is_main_process():
        output_dir.mkdir(exist_ok=True, parents=True)
        slide_features_dir.mkdir(exist_ok=True, parents=True)
        region_features_dir.mkdir(exist_ok=True, parents=True)

    if cfg.level == "global":
        model = GlobalFeatureExtractor(
            region_size=cfg.region_size,
            patch_size=cfg.patch_size,
            mini_patch_size=cfg.mini_patch_size,
            pretrain_vit_patch=cfg.pretrain_vit_patch,
            pretrain_vit_region=cfg.pretrain_vit_region,
            img_size_pretrained=cfg.img_size_pretrained,
            verbose=(gpu_id in [-1, 0]),
        )
    elif cfg.level == "local":
        model = LocalFeatureExtractor(
            patch_size=cfg.patch_size,
            mini_patch_size=cfg.mini_patch_size,
            pretrain_vit_patch=cfg.pretrain_vit_patch,
            verbose=(gpu_id in [-1, 0]),
        )
    else:
        raise ValueError(f"cfg.level ({cfg.level}) not supported")

    assert cfg.csv is not None, "'csv' must be provided"

    df = pd.read_csv(cfg.csv)
    patch_dir = Path(cfg.patch_dir, f"patches/{cfg.region_size}/npy")
    slide_paths = df.slide_path.unique().tolist()
    slide_ids = [Path(s).stem for s in slide_paths if Path(patch_dir, f"{Path(s).stem}.npy").is_file()]

    if is_main_process():
        print(f"{len(slide_ids)} slides with extracted patches found")

    if cfg.slide_list:
        with open(Path(cfg.slide_list), "r") as f:
            slide_ids = sorted([Path(x.strip()).stem for x in f.readlines()])
        if is_main_process():
            print(f"restricting to {len(slide_ids)} slides from slide list .txt file")

    num_workers = min(mp.cpu_count(), cfg.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    if gpu_id == -1:
        device = torch.device(f"cuda")
    else:
        device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device, non_blocking=True)

    if is_main_process():
        print()

    if Path(output_dir, "process_list.csv").is_file() and cfg.resume:
        process_list_fp = Path(output_dir, "process_list.csv")
        process_df = pd.read_csv(process_list_fp)
    else:
        process_df = pd.DataFrame({
            "slide_id": slide_ids,
            "status": ["not processed"] * len(slide_ids),
            "error": [np.nan] * len(slide_ids),
            "feature_path": [np.nan] * len(slide_ids),
        })
        process_df["feature_path"] = process_df["feature_path"].astype(str)

    mask = process_df["status"] == "not processed"
    process_stack = process_df[mask]
    total = len(process_stack)
    already_processed = len(process_df) - total

    slide_ids_to_process = process_stack.slide_id

    sub_df = df[df.slide_id.isin(slide_ids_to_process)].reset_index(drop=True)
    dataset = RegionCoordinatesDataset(sub_df, patch_dir)
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_coordinates,
    )

    processed_count = already_processed

    with tqdm.tqdm(
        loader,
        desc=f"Slide Encoding (GPU: {max(gpu_id, 0)+1}/{torch.cuda.device_count()})",
        unit=" slide",
        unit_scale=1,
        initial=already_processed//torch.cuda.device_count(),
        total=(total+already_processed)//torch.cuda.device_count(),
        leave=True,
        position=max(gpu_id, 0)*2,
    ) as t1:
        with torch.no_grad():
            for batch in t1:

                try:

                    _, wsi_fp, coordinates = batch
                    slide_id = wsi_fp.stem
                    region_dataset = PatchDataset(wsi_fp, coordinates, cfg.backend)
                    region_dataloader = torch.utils.data.DataLoader(region_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
                    region_feature_paths = []
                    with tqdm.tqdm(
                        region_dataloader,
                        desc=f"GPU {max(gpu_id, 0)}: {slide_id}",
                        unit=" region",
                        unit_scale=1,
                        leave=False,
                        position=max(gpu_id, 0)*2+1,
                    ) as t2:
                        for  img, x, y in t2:
                            x, y = x.item(), y.item()
                            img = img.to(device, non_blocking=True)
                            p = None
                            feature = model(img, pct=p)
                            save_path = Path(
                                region_features_dir, f"{slide_id}_{x}_{y}.pt"
                            )
                            torch.save(feature, save_path)
                            region_feature_paths.append(save_path)
                    slide_feature = build_slide_level_feature(region_feature_paths)
                    save_path = Path(slide_features_dir, f"{slide_id}.pt")
                    torch.save(slide_feature, save_path)

                    processed_count += 1

                    mask = process_df["slide_id"] == slide_id
                    process_df.loc[mask, "status"] = "processed"
                    process_df.loc[mask, "error"] = np.nan
                    process_df.loc[mask, "feature_path"] = str(save_path)

                    if cfg.wandb.enable and is_main_process():
                        wandb.log({"processed": processed_count})

                except Exception as e:

                    mask = process_df["slide_id"] == slide_id
                    process_df.loc[mask, "status"] = "error"
                    process_df.loc[mask, "error"] = str(e)

    if is_main_process():
        process_csv_path = Path(output_dir, f"process_list.csv")
        process_df.to_csv(process_csv_path, index=False)


if __name__ == "__main__":
    # python3 -m torch.distributed.run --standalone --nproc_per_node=gpu extract_features.py --config-name 'debug'
    main()
