import os
import time
import wandb
import hydra
import torch
import random
import datetime
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
    test_regression_masked,
    compute_time,
    collate_features,
)


@hydra.main(
    version_base="1.2.0",
    config_path="../config/inference/classification",
    config_name="ecp_masked_attn",
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

    features_root_dir = Path(cfg.features_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    assert (cfg.task != "classification" and cfg.label_encoding != "ordinal") or (
        cfg.task == "classification"
    )

    mask_attention = (cfg.model.mask_attn_patch is True) or (cfg.model.mask_attn_region is True)

    if isinstance(cfg.test_csv, str):
        test_csvs = {"test": cfg.test_csv}
    else:
        test_csvs = {k: v for e in cfg.test_csv for k, v in e.items()}

    checkpoint_root_dir = Path(cfg.model.checkpoints)
    nfold = len([_ for _ in checkpoint_root_dir.glob(f"fold_*")])
    print(f"Found {nfold} models")
    checkpoints = {
        p.name: Path(p, "best.pt")
        for p in sorted(list(checkpoint_root_dir.glob(f"fold_*")))
    }

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
        if not features_dir.exists():
            features_dir = Path(features_root_dir, f"{model_name}")

        model_start_time = time.time()
        for test_name, csv_path in test_csvs.items():
            print(f"Loading {test_name} data")
            test_df = pd.read_csv(csv_path)

            test_dataset_options = ClassificationDatasetOptions(
                df=test_df,
                features_dir=features_dir,
                label_name=cfg.label_name,
                label_mapping=cfg.label_mapping,
                label_encoding=cfg.label_encoding,
                blinded=cfg.blinded,
                num_classes=cfg.num_classes,
                mask_attention=mask_attention,
                region_dir=Path(cfg.region_dir),
                spacing=cfg.spacing,
                region_size=cfg.model.region_size,
                patch_size=cfg.model.patch_size,
                mini_patch_size=cfg.model.mini_patch_size,
                backend=cfg.backend,
                region_format=cfg.region_format,
                segmentation_parameters=cfg.seg_params,
                tissue_pct=cfg.tissue_pct,
            )

            print(f"Initializing test dataset")
            test_dataset = DatasetFactory(cfg.task, test_dataset_options).get_dataset()

            print(f"Running inference on {test_name} dataset with {model_name} model")

            if cfg.task == "regression":
                if mask_attention:
                    test_results = test_regression_masked(
                        model,
                        test_dataset,
                        batch_size=1,
                        num_workers=num_workers,
                        use_wandb=cfg.wandb.enable,
                    )
                else:
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
                    cfg.loss,
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
            test_dataset.df.to_csv(
                Path(result_dir, f"{model_name}_{test_name}.csv"), index=False
            )

            for r, v in test_results.items():
                if r == "auc":
                    v = round(v, 5)
                if r == "cm":
                    save_path = Path(result_dir, f"{model_name}_{test_name}.png")
                    v.savefig(save_path, bbox_inches="tight")
                    plt.close(v)
                if cfg.wandb.enable and r in log_to_wandb["test"]:
                    if r == "cm":
                        wandb.log(
                            {
                                f"{test_name}/{model_name}/{r}": wandb.Image(
                                    str(save_path)
                                )
                            }
                        )
                    else:
                        wandb.log({f"{test_name}/{model_name}/{r}": v})
                elif "cm" not in r:
                    print(f"{model_name} ({test_name}): {r} = {v}")

        model_end_time = time.time()
        mins, secs = compute_time(model_start_time, model_end_time)
        print(f"Time taken ({model_name}): {mins}m {secs}s")
        print()

    distance_func = None
    if cfg.distance_func == "custom":
        distance_func = custom_isup_grade_dist
    for test_name, csv_path in test_csvs.items():
        dfs = []
        cols = ["slide_id", "label", "pred"]
        for model_name in checkpoints:
            df = pd.read_csv(Path(result_dir, f"{model_name}_{test_name}.csv"))[cols]
            df = df.rename(columns={"pred": f"pred_{model_name}"})
            dfs.append(df)
        ensemble_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=["slide_id", "label"], how="outer"
            ),
            dfs,
        )
        ensemble_df["agg"] = ensemble_df[
            [f"pred_{model_name}" for model_name in checkpoints]
        ].apply(lambda x: get_majority_vote(x, distance_func, seed=x.name), axis=1)

        test_df = pd.read_csv(csv_path)
        missing_sids = set(test_df.slide_id.values).difference(
            set(ensemble_df.slide_id.values)
        )
        missing_df = pd.DataFrame.from_dict(
            {
                "slide_id": list(missing_sids),
                "label": test_df[test_df.slide_id.isin(missing_sids)][
                    f"{cfg.label_name}"
                ].values.tolist(),
                "agg": [
                    random.randint(0, cfg.num_classes - 1)
                    for _ in range(len(missing_sids))
                ],
            }
        )
        ensemble_df = pd.concat([ensemble_df, missing_df], ignore_index=True)
        ensemble_df.to_csv(Path(result_dir, f"{test_name}.csv"), index=False)
        ensemble_metrics = get_metrics(
            ensemble_df["agg"].values,
            ensemble_df.label.values,
            class_names=[f"isup_{i}" for i in range(cfg.num_classes)],
            use_wandb=cfg.wandb.enable,
        )

        for r, v in ensemble_metrics.items():
            if isinstance(v, float):
                v = round(v, 5)
            if r == "cm":
                save_path = Path(result_dir, f"ensemble_{test_name}.png")
                v.savefig(save_path, bbox_inches="tight")
                plt.close(v)
            if cfg.wandb.enable and r in log_to_wandb["test"]:
                if r == "cm":
                    wandb.log(
                        {f"{test_name}/ensemble_{r}": wandb.Image(str(save_path))}
                    )
                else:
                    wandb.log({f"{test_name}/ensemble_{r}": v})
            elif "cm" not in r:
                print(f"{test_name} ensemble: {r} = {v}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

    main()
