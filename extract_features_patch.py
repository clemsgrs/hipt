import os
import tqdm
import time
import wandb
import torch
import hydra
import shutil
import datetime
import subprocess
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from omegaconf import DictConfig

import source.distributed as distributed
from source.dataset import ImageFolderWithNameDataset
from source.models import PatchEmbedder
from source.utils import (
    initialize_wandb,
    is_main_process,
)
from pretrain.utils import make_classification_eval_transform


@hydra.main(
    version_base="1.2.0", config_path="config/feature_extraction", config_name="patch"
)
def main(cfg: DictConfig):
    
    distributed.setup_distributed()
    distributed.fix_random_seed(cfg.seed)

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

    if distributed.is_enabled_and_multiple_gpus():
        obj = [run_id]
        torch.distributed.broadcast_object_list(
            obj, 0, device=torch.device(f"cuda:{distributed.get_global_rank()}")
        )
        run_id = obj[0]

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    features_dir = Path(output_dir, "features")
    if not cfg.resume and is_main_process():
        if output_dir.exists():
            print(f"{output_dir} already exists! deleting it...")
            shutil.rmtree(output_dir)
            print("done")
            output_dir.mkdir(parents=False)
            features_dir.mkdir()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            features_dir.mkdir(exist_ok=True)

    model = PatchEmbedder(
        img_size=cfg.patch_size,
        mini_patch_size=cfg.mini_patch_size,
        pretrain_vit_patch=cfg.pretrain_vit_patch,
        verbose=(distributed.get_global_rank() in [-1, 0]),
        img_size_pretrained=cfg.img_size_pretrained,
    )

    transform = make_classification_eval_transform(img_size=cfg.patch_size)
    dataset = ImageFolderWithNameDataset(cfg.data_dir, transform)

    if distributed and is_main_process() and cfg.wandb.enable:
        command_line = [
            "python3",
            "log_nproc.py",
            "--output_dir",
            f"{features_dir}",
            "--fmt",
            "pt",
            "--total",
            f"{len(dataset)}",
            "--log_to_wandb",
            "--id",
            f"{run_id}",
        ]
        subprocess.Popen(command_line)

    time.sleep(5)

    if distributed.is_enabled_and_multiple_gpus():
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    num_workers = min(mp.cpu_count(), cfg.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device(f"cuda:{distributed.get_global_rank()}")
    model = model.to(device, non_blocking=True)

    if is_main_process():
        print()

    filenames, feature_paths = [], []

    with tqdm.tqdm(
        loader,
        desc="Feature Extraction",
        unit=" img",
        ncols=80,
        unit_scale=cfg.batch_size,
        position=0,
        leave=True,
        disable=not (distributed.get_global_rank() in [-1, 0]),
    ) as t1:
        with torch.no_grad():
            for i, batch in enumerate(t1):
                imgs, fnames = batch
                imgs = imgs.to(device, non_blocking=True)
                features = model(imgs)
                for k, f in enumerate(features):
                    fname = fnames[k]
                    feature_path = Path(
                        features_dir, f"{fname}.pt"
                    )
                    torch.save(f, feature_path)
                    filenames.append(fname)
                    feature_paths.append(feature_path)
                if cfg.wandb.enable and not distributed.is_enabled_and_multiple_gpus():
                    wandb.log({"processed": i + imgs.shape[0]})

    features_df = pd.DataFrame.from_dict(
        {
            "filename": filenames,
            "feature_path": feature_paths,
        }
    )

    if distributed.is_enabled_and_multiple_gpus():
        features_csv_path = Path(output_dir, f"features_{distributed.get_global_rank()}.csv")
    else:
        features_csv_path = Path(output_dir, f"features.csv")
    features_df.to_csv(features_csv_path, index=False)

    if distributed.is_enabled_and_multiple_gpus():
        torch.distributed.barrier()
        if is_main_process():
            dfs = []
            for distributed.get_global_rank() in range(torch.cuda.device_count()):
                fp = Path(output_dir, f"features_{distributed.get_global_rank()}.csv")
                df = pd.read_csv(fp)
                dfs.append(df)
                os.remove(fp)
            features_df = pd.concat(dfs, ignore_index=True)
            features_df = features_df.drop_duplicates()
            features_df.to_csv(
                Path(output_dir, f"features.csv"), index=False
            )

    if cfg.wandb.enable and is_main_process() and distributed.is_enabled_and_multiple_gpus():
        wandb.log({"processed": len(features_df)})


if __name__ == "__main__":
    # python3 -m torch.distributed.run --standalone --nproc_per_node=gpu extract_features_patch.py --config-name 'patch'
    main()
