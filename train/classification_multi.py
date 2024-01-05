import os
import time
import tqdm
import wandb
import torch
import hydra
import datetime
import matplotlib
import statistics
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict

from source.models import ModelFactory
from source.components import LossFactory
from source.dataset import ClassificationDatasetOptions, DatasetFactory
from source.augmentations import AugmentationOptions, FeatureSpaceAugmentation
from source.utils import (
    initialize_wandb,
    train,
    train_ordinal,
    train_regression,
    train_regression_masked,
    tune,
    tune_ordinal,
    tune_regression,
    tune_regression_masked,
    test,
    test_ordinal,
    test_regression,
    test_regression_masked,
    compute_time,
    update_log_dict,
    collate_features,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)


@hydra.main(
    version_base="1.2.0",
    config_path="../config/training/classification",
    config_name="multi",
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

    checkpoint_root_dir = Path(output_dir, "checkpoints")
    checkpoint_root_dir.mkdir(parents=True, exist_ok=True)

    result_root_dir = Path(output_dir, "results")
    result_root_dir.mkdir(parents=True, exist_ok=True)

    if cfg.augmentation.use:
        aug_root_dir = Path(output_dir, "augmentation")
        aug_root_dir.mkdir(parents=True, exist_ok=True)

    features_root_dir = Path(cfg.features_root_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    assert (cfg.task != "classification" and cfg.label_encoding != "ordinal") or (
        cfg.task == "classification"
    )

    mask_attn = (cfg.model.mask_attn_patch is True) or (cfg.model.mask_attn_region is True)

    fold_root_dir = Path(cfg.data.fold_dir)
    nfold = len([_ for _ in fold_root_dir.glob(f"fold_*")])
    print(f"Training on {nfold} folds")

    tune_metrics = defaultdict(dict)
    test_metrics = defaultdict(dict)

    start_time = time.time()
    for i in range(nfold):
        fold_dir = Path(fold_root_dir, f"fold_{i}")
        checkpoint_dir = Path(checkpoint_root_dir, f"fold_{i}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir = Path(result_root_dir, f"fold_{i}")
        result_dir.mkdir(parents=True, exist_ok=True)

        if cfg.data.fold_specific_features:
            slide_features_dir = Path(features_root_dir, f"fold_{i}/slide_features")
            region_features_dir = Path(features_root_dir, f"fold_{i}/region_features")
        else:
            slide_features_dir = Path(features_root_dir, f"slide_features")
            region_features_dir = Path(features_root_dir, f"region_features")

        print(f"Loading data for fold {i+1}")
        train_df_path = Path(fold_dir, "train.csv")
        tune_df_path = Path(fold_dir, "tune.csv")
        test_df_path = Path(fold_dir, "test.csv")
        train_df = pd.read_csv(train_df_path)
        tune_df = pd.read_csv(tune_df_path)
        if test_df_path.is_file():
            test_df = pd.read_csv(test_df_path)

        if cfg.training.pct:
            print(f"Training & tuning on {cfg.training.pct*100}% of the data")
            train_df = train_df.sample(frac=cfg.training.pct).reset_index(drop=True)
            tune_df = tune_df.sample(frac=cfg.training.pct).reset_index(drop=True)

        transform = None
        if cfg.augmentation.use:
            aug_dir = Path(aug_root_dir, f"fold_{i}")
            aug_dir.mkdir(parents=True, exist_ok=True)
            csv_path = Path(region_features_dir.parent, "region_features.csv")
            if csv_path.is_file():
                region_df = pd.read_csv(csv_path)
            elif cfg.augmentation.name in ["interpolation", "extrapolation"]:
                raise OSError(f"{csv_path} doesn't exist!")
            else:
                region_df = None
            kwargs = {k: v for e in cfg.augmentation.kwargs for k, v in e.items()}
            aug_options = AugmentationOptions(
                name=cfg.augmentation.name,
                output_dir=aug_dir,
                region_features_dir=region_features_dir,
                region_df=region_df,
                label_df=train_df,
                label_name=cfg.label_name,
                level=cfg.level,
                multiprocessing=(num_workers == 0),
                kwargs=kwargs,
            )
            transform = FeatureSpaceAugmentation(aug_options)

        train_dataset_options = ClassificationDatasetOptions(
            df=train_df,
            features_dir=slide_features_dir,
            label_name=cfg.label_name,
            label_mapping=cfg.label_mapping,
            label_encoding=cfg.label_encoding,
            transform=transform,
            mask_attention=mask_attn,
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
        tune_dataset_options = ClassificationDatasetOptions(
            df=tune_df,
            features_dir=slide_features_dir,
            label_name=cfg.label_name,
            label_mapping=cfg.label_mapping,
            label_encoding=cfg.label_encoding,
            mask_attention=mask_attn,
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
        if test_df_path.is_file():
            test_dataset_options = ClassificationDatasetOptions(
                df=test_df,
                features_dir=slide_features_dir,
                label_name=cfg.label_name,
                label_mapping=cfg.label_mapping,
                label_encoding=cfg.label_encoding,
                mask_attention=mask_attn,
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

        print(f"Initializing datasets")
        train_dataset = DatasetFactory(cfg.task, train_dataset_options).get_dataset()
        tune_dataset = DatasetFactory(cfg.task, tune_dataset_options).get_dataset()
        if test_df_path.is_file():
            test_dataset = DatasetFactory(cfg.task, test_dataset_options).get_dataset()

        m, n = train_dataset.num_classes, tune_dataset.num_classes
        assert (
            m == n == cfg.num_classes
        ), f"Either train (C={m}) or tune (C={n}) sets doesnt cover full class spectrum (C={cfg.num_classes}"

        criterion = LossFactory(
            cfg.task, cfg.loss, cfg.label_encoding, cfg.loss_options
        ).get_loss()

        model = ModelFactory(
            cfg.level,
            cfg.num_classes,
            cfg.task,
            cfg.loss,
            cfg.label_encoding,
            cfg.model,
        ).get_model()
        model.relocate()
        print(model)

        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = OptimizerFactory(
            cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
        ).get_optimizer()
        scheduler = SchedulerFactory(optimizer, cfg.optim.lr_scheduler).get_scheduler()

        early_stopping = EarlyStopping(
            cfg.early_stopping.tracking,
            cfg.early_stopping.min_max,
            cfg.early_stopping.patience,
            cfg.early_stopping.min_epoch,
            checkpoint_dir=checkpoint_dir,
            save_all=cfg.early_stopping.save_all,
        )

        stop = False
        fold_start_time = time.time()

        if cfg.wandb.enable:
            wandb.define_metric(f"train/fold_{i}/epoch", summary="max")

        with tqdm.tqdm(
            range(cfg.nepochs),
            desc=(f"HIPT Training (fold {i+1}/{nfold})"),
            unit=" epoch",
            ncols=100,
            leave=True,
        ) as t:
            for epoch in t:

                # set dataset seed
                train_dataset.seed = epoch

                epoch_start_time = time.time()
                if cfg.wandb.enable:
                    log_dict = {f"train/fold_{i}/epoch": epoch + 1}

                if cfg.task == "regression":
                    if mask_attn:
                        train_results = train_regression_masked(
                            epoch + 1,
                            model,
                            train_dataset,
                            optimizer,
                            criterion,
                            batch_size=cfg.training.batch_size,
                            weighted_sampling=cfg.training.weighted_sampling,
                            gradient_accumulation=cfg.training.gradient_accumulation,
                            num_workers=num_workers,
                            use_wandb=cfg.wandb.enable,
                        )
                    else:
                        train_results = train_regression(
                            epoch + 1,
                            model,
                            train_dataset,
                            optimizer,
                            criterion,
                            batch_size=cfg.training.batch_size,
                            weighted_sampling=cfg.training.weighted_sampling,
                            gradient_accumulation=cfg.training.gradient_accumulation,
                            num_workers=num_workers,
                            use_wandb=cfg.wandb.enable,
                        )
                elif cfg.label_encoding == "ordinal":
                    train_results = train_ordinal(
                        epoch + 1,
                        model,
                        train_dataset,
                        optimizer,
                        criterion,
                        cfg.loss,
                        batch_size=cfg.training.batch_size,
                        weighted_sampling=cfg.training.weighted_sampling,
                        gradient_accumulation=cfg.training.gradient_accumulation,
                        num_workers=num_workers,
                        use_wandb=cfg.wandb.enable,
                    )
                else:
                    train_results = train(
                        epoch + 1,
                        model,
                        train_dataset,
                        optimizer,
                        criterion,
                        collate_fn=partial(collate_features, label_type="int"),
                        batch_size=cfg.training.batch_size,
                        weighted_sampling=cfg.training.weighted_sampling,
                        gradient_accumulation=cfg.training.gradient_accumulation,
                        num_workers=num_workers,
                        use_wandb=cfg.wandb.enable,
                    )

                if cfg.wandb.enable:
                    update_log_dict(
                        f"train/fold_{i}",
                        train_results,
                        log_dict,
                        step=f"train/fold_{i}/epoch",
                        to_log=log_to_wandb["train"],
                    )
                for r, v in train_results.items():
                    if isinstance(v, matplotlib.figure.Figure):
                        plt.close(v)
                train_dataset.df.to_csv(
                    Path(result_dir, f"train_{epoch+1}.csv"), index=False
                )

                if epoch % cfg.tuning.tune_every == 0:
                    if cfg.task == "regression":
                        if mask_attn:
                            tune_results = tune_regression_masked(
                                epoch + 1,
                                model,
                                tune_dataset,
                                criterion,
                                batch_size=cfg.tuning.batch_size,
                                num_workers=num_workers,
                                use_wandb=cfg.wandb.enable,
                            )
                        else:
                            tune_results = tune_regression(
                                epoch + 1,
                                model,
                                tune_dataset,
                                criterion,
                                batch_size=cfg.tuning.batch_size,
                                num_workers=num_workers,
                                use_wandb=cfg.wandb.enable,
                            )
                    elif cfg.label_encoding == "ordinal":
                        tune_results = tune_ordinal(
                            epoch + 1,
                            model,
                            tune_dataset,
                            criterion,
                            cfg.loss,
                            batch_size=cfg.tuning.batch_size,
                            num_workers=num_workers,
                            use_wandb=cfg.wandb.enable,
                        )
                    else:
                        tune_results = tune(
                            epoch + 1,
                            model,
                            tune_dataset,
                            criterion,
                            collate_fn=partial(collate_features, label_type="int"),
                            batch_size=cfg.tuning.batch_size,
                            num_workers=num_workers,
                            use_wandb=cfg.wandb.enable,
                        )

                    if cfg.wandb.enable:
                        update_log_dict(
                            f"tune/fold_{i}",
                            tune_results,
                            log_dict,
                            step=f"train/fold_{i}/epoch",
                            to_log=[e for e in log_to_wandb["tune"] if "cm" not in e],
                        )
                    for r, v in tune_results.items():
                        if isinstance(v, matplotlib.figure.Figure):
                            plt.close(v)
                    tune_dataset.df.to_csv(
                        Path(result_dir, f"tune_{epoch+1}.csv"), index=False
                    )

                    early_stopping(epoch, model, tune_results)
                    if early_stopping.early_stop and cfg.early_stopping.enable:
                        stop = True

                lr = cfg.optim.lr
                if scheduler:
                    lr = scheduler.get_last_lr()[0]
                    scheduler.step()
                if cfg.wandb.enable:
                    wandb.define_metric(
                        f"train/fold_{i}/lr", step_metric=f"train/fold_{i}/epoch"
                    )
                    log_dict.update({f"train/fold_{i}/lr": lr})

                if cfg.wandb.enable:
                    wandb.log(log_dict)

                epoch_end_time = time.time()
                epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
                tqdm.tqdm.write(
                    f"End of epoch {epoch+1} / {cfg.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
                )

                if stop:
                    tqdm.tqdm.write(
                        f"Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago"
                    )
                    break

        fold_end_time = time.time()
        fold_mins, fold_secs = compute_time(fold_start_time, fold_end_time)
        print(f"Total time taken for fold {i+1}/{nfold}: {fold_mins}m {fold_secs}s")

        # load best model
        best_model_fp = Path(checkpoint_dir, f"best.pt")
        if cfg.wandb.enable:
            wandb.save(str(best_model_fp))
        best_model_sd = torch.load(best_model_fp)
        model.load_state_dict(best_model_sd)

        if cfg.task == "regression":
            if mask_attn:
                tune_results = test_regression_masked(
                    model,
                    tune_dataset,
                    batch_size=1,
                    num_workers=num_workers,
                    use_wandb=cfg.wandb.enable,
                )
            else:
                tune_results = test_regression(
                    model,
                    tune_dataset,
                    batch_size=1,
                    num_workers=num_workers,
                    use_wandb=cfg.wandb.enable,
                )
        elif cfg.label_encoding == "ordinal":
            tune_results = test_ordinal(
                model,
                tune_dataset,
                cfg.loss,
                batch_size=1,
                num_workers=num_workers,
                use_wandb=cfg.wandb.enable,
            )
        else:
            tune_results = test(
                model,
                tune_dataset,
                collate_fn=partial(collate_features, label_type="int"),
                batch_size=1,
                num_workers=num_workers,
                use_wandb=cfg.wandb.enable,
            )
        tune_dataset.df.to_csv(
            Path(result_dir, f"tune_{cfg.testing.retrieve_checkpoint}.csv"), index=False
        )

        for r, v in tune_results.items():
            tune_metrics[f"fold_{i}"][r] = v
            if isinstance(v, float):
                v = round(v, 5)
            if r == "cm":
                save_path = Path(
                    result_dir, f"tune_{cfg.testing.retrieve_checkpoint}_cm.png"
                )
                v.savefig(save_path, bbox_inches="tight")
                plt.close(v)
            if cfg.wandb.enable and r in log_to_wandb["tune"]:
                if r == "cm":
                    wandb.log(
                        {
                            f"tune/fold_{i}/{r}_{cfg.testing.retrieve_checkpoint}": wandb.Image(
                                str(save_path)
                            )
                        }
                    )
                else:
                    wandb.log(
                        {f"tune/fold_{i}/{r}_{cfg.testing.retrieve_checkpoint}": v}
                    )
            elif "cm" not in r:
                print(f"tune {r}_{cfg.testing.retrieve_checkpoint}: {v}")

        if test_df_path.is_file():
            if cfg.task == "regression":
                if mask_attn:
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
            test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)

            for r, v in test_results.items():
                test_metrics[f"fold_{i}"][r] = v
                if isinstance(v, float):
                    v = round(v, 5)
                if r == "cm":
                    save_path = Path(result_dir, f"test_cm.png")
                    v.savefig(save_path, bbox_inches="tight")
                    plt.close(v)
                if cfg.wandb.enable and r in log_to_wandb["test"]:
                    if r == "cm":
                        wandb.log({f"test/fold_{i}/{r}": wandb.Image(str(save_path))})
                    else:
                        wandb.log({f"test/fold_{i}/{r}": v})
                elif "cm" not in r:
                    print(f"test {r}: {v}")

    metrics = defaultdict(list)
    for _, metric_dict in tune_metrics.items():
        for metric_name, metric_val in metric_dict.items():
            if isinstance(metric_val, float):
                metrics[metric_name].append(metric_val)

    mean_tune_metrics = {
        metric_name: round(np.mean(metric_values), 5)
        for metric_name, metric_values in metrics.items()
    }
    std_tune_metrics = {
        metric_name: round(statistics.stdev(metric_values), 5)
        for metric_name, metric_values in metrics.items()
    }
    for name in metrics.keys():
        mean = mean_tune_metrics[name]
        std = std_tune_metrics[name]
        if cfg.wandb.enable:
            wandb.log({f"tune/{name}_mean": mean})
            wandb.log({f"tune/{name}_std": std})
        else:
            print(f"mean tune {name}: {mean} ± {std}")

    metrics = defaultdict(list)
    for _, metric_dict in test_metrics.items():
        for metric_name, metric_val in metric_dict.items():
            if isinstance(metric_val, float):
                metrics[metric_name].append(metric_val)

    mean_test_metrics = {
        metric_name: round(np.mean(metric_values), 5)
        for metric_name, metric_values in metrics.items()
    }
    std_test_metrics = {
        metric_name: round(statistics.stdev(metric_values), 5)
        for metric_name, metric_values in metrics.items()
    }
    for name in metrics.keys():
        mean = mean_test_metrics[name]
        std = std_test_metrics[name]
        if cfg.wandb.enable:
            wandb.log({f"test/{name}_mean": mean})
            wandb.log({f"test/{name}_std": std})
        else:
            print(f"mean test {name}: {mean} ± {std}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken ({nfold} folds): {mins}m {secs}s")


if __name__ == "__main__":

    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

    main()
