import os
import time
import wandb
import hydra
import torch
import datetime
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from omegaconf import DictConfig

from source.models import ModelFactory
from source.dataset import ClassificationDatasetOptions, DatasetFactory
from source.utils import (
    initialize_wandb,
    test,
    test_ordinal,
    test_regression,
    compute_time,
    collate_features,
)


@hydra.main(
    version_base="1.2.0",
    config_path="../config/inference/classification",
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

    features_root_dir = Path(cfg.features_root_dir)
    slide_features_dir = Path(features_root_dir, f"slide_features")

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    assert (cfg.task != "classification" and cfg.label_encoding != "ordinal") or (
        cfg.task == "classification"
    )

    model = ModelFactory(
        cfg.level, cfg.num_classes, cfg.task, cfg.loss, cfg.label_encoding, cfg.model
    ).get_model()
    model.relocate()
    print(model)
    print()

    print(f"Loading provided model checkpoint")
    print(cfg.model.checkpoint)
    sd = torch.load(cfg.model.checkpoint)
    msg = model.load_state_dict(sd)
    print(f"Checkpoint loaded with msg: {msg}")

    test_csvs = {k: v for e in cfg.test_csv for k, v in e.items()}
    start_time = time.time()
    for test_name, csv_path in test_csvs.items():
        print(f"Loading {test_name} data")
        test_df = pd.read_csv(csv_path)

        test_dataset_options = ClassificationDatasetOptions(
            df=test_df,
            features_dir=slide_features_dir,
            label_name=cfg.label_name,
            label_mapping=cfg.label_mapping,
            label_encoding=cfg.label_encoding,
        )

        print(f"Initializing test dataset")
        test_dataset = DatasetFactory(cfg.task, test_dataset_options).get_dataset()

        print(f"Running inference on {test_name} dataset")

        if cfg.task == "regression":
            test_results = test_regression(
                model,
                test_dataset,
                batch_size=1,
                num_workers=num_workers,
                use_wandb=cfg.wandb.enable,
            )
        elif cfg.label_encoding == "ordinal":
            test_results = test_ordinal(
                model,
                test_dataset,
                batch_size=1,
                num_workers=num_workers,
                use_wandb=cfg.wandb.enable,
            )
        else:
            test_results = test(
                model,
                test_dataset,
                collate_fn=partial(collate_features, label_type="int"),
                batch_size=1,
                num_workers=num_workers,
                use_wandb=cfg.wandb.enable,
            )
        test_dataset.df.to_csv(Path(result_dir, f"{test_name}.csv"), index=False)
        print()

        for r, v in test_results.items():
            if r == "auc":
                v = round(v, 5)
            if r == "cm":
                save_path = Path(result_dir, f"{test_name}.png")
                v.savefig(save_path, bbox_inches="tight")
                plt.close(v)
            if r in log_to_wandb["test"] and cfg.wandb.enable:
                if r == "cm":
                    wandb.log({f"{test_name}/{r}": wandb.Image(str(save_path))})
                else:
                    wandb.log({f"{test_name}/{r}": v})
            elif "cm" not in r:
                print(f"{test_name} {r}: {v}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy("file_system")

    main()
