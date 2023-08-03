import os
import time
import wandb
import hydra
import torch
import datetime
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from omegaconf import DictConfig

from source.models import ModelFactory
from source.dataset import (
    SurvivalDatasetOptions,
    DatasetFactory,
    ppcess_survival_data,
    ppcess_tcga_survival_data,
)
from source.utils import (
    initialize_wandb,
    test_survival,
    compute_time,
)


@hydra.main(
    version_base="1.2.0",
    config_path="../config/inference/survival",
    config_name="default",
)
def main(cfg: DictConfig):
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        log_to_wandb = {k: v for e in cfg.wandb.to_log for k, v in e.items()}
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(output_dir, "checkpoints", cfg.level)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(output_dir, "results", cfg.level)
    result_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cfg.features_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    tiles_df = None
    if (
        cfg.model.slide_pos_embed.type == "2d"
        and cfg.model.slide_pos_embed.use
        and cfg.model.agg_method
    ):
        tiles_df = pd.read_csv(cfg.data.tiles_csv)

    print("Loading test data")
    test_df = pd.read_csv(cfg.test_csv)
    patient_df, slide_df = ppcess_survival_data(
        test_df, cfg.label_name, nbins=cfg.nbins
    )

    test_dataset_options = SurvivalDatasetOptions(
        patient_df=patient_df,
        slide_df=slide_df,
        tiles_df=tiles_df,
        features_dir=features_dir,
        label_name=cfg.label_name,
    )

    print(f"Initializing test dataset")
    test_dataset = DatasetFactory(
        "survival", test_dataset_options, cfg.model.agg_method
    ).get_dataset()

    model = ModelFactory(
        cfg.level,
        num_classes=cfg.nbins,
        task="survival",
        loss=cfg.loss,
        label_encoding=cfg.label_encoding,
        model_options=cfg.model,
    ).get_model()
    model.relocate()
    print(model)

    print(f"Loading provided model checkpoint")
    print(cfg.model.checkpoint)
    sd = torch.load(cfg.model.checkpoint)
    msg = model.load_state_dict(sd)
    print(f"Checkpoint loaded with msg: {msg}")

    print(f"Running inference on test dataset")
    start_time = time.time()

    test_results = test_survival(
        model,
        test_dataset,
        agg_method=cfg.model.agg_method,
        batch_size=1,
        num_workers=num_workers,
    )
    test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)

    for r, v in test_results.items():
        if isinstance(v, float):
            v = round(v, 3)
        if r in log_to_wandb["test"] and cfg.wandb.enable:
            wandb.log({f"test/{r}": v})
        elif "cm" not in r:
            print(f"Test {r}: {v}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":
    main()
