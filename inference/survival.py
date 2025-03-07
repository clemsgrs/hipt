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
)
from source.utils import (
    initialize_wandb,
    test_survival,
    compute_time,
    update_state_dict,
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

    result_dir = Path(output_dir, "results")
    result_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cfg.features_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    if isinstance(cfg.test_csv, str):
        test_csvs = {"test": cfg.test_csv}
    else:
        test_csvs = {k: v for e in cfg.test_csv for k, v in e.items()}
    start_time = time.time()
    for test_name, csv_path in test_csvs.items():
        print(f"Loading {test_name} data")
        test_df = pd.read_csv(csv_path)

        test_dataset_options = SurvivalDatasetOptions(
            phase="test",
            df=test_df,
            features_dir=features_dir,
            label_name=cfg.label_name,
            nfeats_max=cfg.model.nfeats_max,
        )

        print(f"Initializing test dataset")
        test_dataset = DatasetFactory("survival", test_dataset_options).get_dataset()

        model = ModelFactory(
            cfg.architecture,
            cfg.level,
            num_classes=cfg.nbins,
            task="survival",
            label_encoding=cfg.label_encoding,
            model_options=cfg.model,
        ).get_model()
        model.relocate()
        print(model)

        print(f"Pretrained weights found at {cfg.model.checkpoint}")
        sd = torch.load(cfg.model.checkpoint)
        state_dict, msg = update_state_dict(model.state_dict(), sd)
        model.load_state_dict(state_dict, strict=False)
        print(msg)

        print(f"Running inference on test dataset")
        start_time = time.time()

        test_results = test_survival(
            model,
            test_dataset,
            agg_method=cfg.model.agg_method,
            batch_size=1,
            num_workers=num_workers,
        )
        test_dataset.df.to_csv(Path(result_dir, f"predictions.csv"), index=False)

        for r, v in test_results.items():
            if isinstance(v, float):
                v = round(v, 3)
            if cfg.wandb.enable and r in log_to_wandb["test"]:
                wandb.log({f"test/{r}": v})
            elif "cm" not in r:
                print(f"Test {r}: {v}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":
    main()
