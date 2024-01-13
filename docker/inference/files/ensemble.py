import os
import time
import wandb
import hydra
import torch
import random
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial, reduce
from omegaconf import DictConfig

from source.models import ModelFactory
from source.dataset import ClassificationDatasetOptions, DatasetFactory
from source.utils import (
    initialize_wandb,
    get_metrics,
    get_majority_vote,
    custom_isup_grade_dist,
    test,
    test_ordinal,
    test_regression,
    compute_time,
    collate_features,
)


@hydra.main(
    version_base="1.2.0",
    config_path="../config/inference",
    config_name="ensemble",
)
def main(cfg: DictConfig):

    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        log_to_wandb = {k: v for e in cfg.wandb.to_log for k, v in e.items()}
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(output_dir, "results")
    result_dir.mkdir(parents=True, exist_ok=True)

    features_root_dir = Path(cfg.features_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    assert (cfg.task != "classification" and cfg.label_encoding != "ordinal") or (
        cfg.task == "classification"
    )

    if isinstance(cfg.test_csv, str):
        test_csvs = {'test': cfg.test_csv}
    else:
        test_csvs = {k: v for e in cfg.test_csv for k, v in e.items()}

    checkpoint_root_dir = Path(cfg.model.checkpoints)
    checkpoints = {
        p.stem: p
        for p in sorted(list(checkpoint_root_dir.glob(f"*.pt")))
    }
    nfold = len(checkpoints)
    print(f"Found {nfold} models")

    start_time = time.time()
    for i, (model_name, checkpoint_path) in enumerate(checkpoints.items()):
        model = ModelFactory(
            cfg.level,
            cfg.num_classes,
            cfg.task,
            cfg.loss,
            cfg.label_encoding,
            cfg.model,
        ).get_model()
        model.relocate()
        if i == 0:
            print(model)
            print()

        print(f"Loading {model_name} checkpoint")
        print(checkpoint_path)
        sd = torch.load(checkpoint_path)
        msg = model.load_state_dict(sd)
        print(f"Checkpoint loaded with msg: {msg}")

        features_dir = Path(features_root_dir, f"{model_name}", "slide_features")

        model_start_time = time.time()
        for test_name, csv_path in test_csvs.items():
            print(f"Loading {test_name} data")
            test_df = pd.read_csv(csv_path)

            test_dataset_options = ClassificationDatasetOptions(
                df=test_df,
                features_dir=features_dir,
                blinded=True,
                num_classes=cfg.num_classes,
            )

            print(f"Initializing test dataset")
            test_dataset = DatasetFactory(cfg.task, test_dataset_options).get_dataset()

            print(f"Running inference on {test_name} dataset with {model_name} model")

            if cfg.task == "regression":
                test_regression(
                    model,
                    test_dataset,
                    batch_size=1,
                    num_workers=num_workers,
                    use_wandb=cfg.wandb.enable,
                )
            elif cfg.label_encoding == "ordinal":
                test_ordinal(
                    model,
                    test_dataset,
                    cfg.loss,
                    batch_size=1,
                    num_workers=num_workers,
                    use_wandb=cfg.wandb.enable,
                )
            else:
                test(
                    model,
                    test_dataset,
                    collate_fn=partial(collate_features, label_type="int"),
                    batch_size=1,
                    num_workers=num_workers,
                    use_wandb=cfg.wandb.enable,
                )
            test_dataset.df.to_csv(
                Path(result_dir, f"{model_name}_{test_name}.csv"), index=False
            )

        model_end_time = time.time()
        mins, secs = compute_time(model_start_time, model_end_time)
        print(f"Time taken ({model_name}): {mins}m {secs}s")
        print()

    distance_func = None
    if cfg.distance_func == 'custom':
        distance_func = custom_isup_grade_dist
    for test_name, csv_path in test_csvs.items():
        dfs = []
        cols = ["slide_id", "pred"]
        for model_name in checkpoints:
            df = pd.read_csv(Path(result_dir, f"{model_name}_{test_name}.csv"))[cols]
            df = df.rename(columns={"pred": f"pred_{model_name}"})
            dfs.append(df)
        ensemble_df = reduce(
            lambda left, right: pd.merge(
                left, right, on="slide_id", how="outer"
            ),
            dfs,
        )
        ensemble_df["pred"] = ensemble_df[
            [f"pred_{model_name}" for model_name in checkpoints]
        ].apply(lambda x: get_majority_vote(x, distance_func, seed=x.name), axis=1)

        test_df = pd.read_csv(csv_path)
        missing_sids = set(test_df.slide_id.values).difference(
            set(ensemble_df.slide_id.values)
        )
        missing_df = pd.DataFrame.from_dict(
            {
                "slide_id": list(missing_sids),
                "pred": [
                    random.randint(0, cfg.num_classes - 1)
                    for _ in range(len(missing_sids))
                ],
            }
        )
        ensemble_df = pd.concat([ensemble_df, missing_df], ignore_index=True)
        ensemble_df.to_csv(Path(result_dir, f"submission.csv"), index=False)

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy("file_system")

    main()
