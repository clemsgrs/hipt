import argparse
import gc
import os
import time
import tqdm
import wandb
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from collections import defaultdict

from src.models import ModelFactory
from src.utils import (
    setup,
    train_classification as train,
    tune_classification as tune,
    inference_classification as inference,
    LossFactory,
    OptimizerFactory,
    SchedulerFactory,
    EarlyStopping,
    compute_time,
    update_log_dict,
)
from src.data.dataset import DatasetOptions, ExtractedFeaturesDataset


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("hipt", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    return parser


def main(args):

    config_file = args.config_file
    cfg = setup(config_file)

    output_dir = Path(cfg.output_dir)

    checkpoint_root_dir = output_dir / "checkpoints"
    checkpoint_root_dir.mkdir(parents=True, exist_ok=True)

    result_root_dir = output_dir / "results"
    result_root_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cfg.features_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    fold_root_dir = Path(cfg.data.fold_dir)
    nfold = len([_ for _ in fold_root_dir.glob("fold*")])
    print(f"Training on {nfold} folds")
    print()

    tune_metrics = defaultdict(dict)
    test_metrics = defaultdict(dict)

    start_time = time.time()
    for i in range(nfold):
        fold_dir = Path(fold_root_dir, f"fold-{i}")
        if not fold_dir.is_dir():
            fold_dir = Path(fold_root_dir, f"fold_{i}")
            assert fold_dir.is_dir()
        checkpoint_dir = Path(checkpoint_root_dir, f"fold-{i}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir = Path(result_root_dir, f"fold-{i}")
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading data for fold {i+1}")

        train_df_path = Path(fold_dir, "train.csv")
        train_df = pd.read_csv(train_df_path)

        tune_df_path = Path(fold_dir, "tune.csv")
        tune_df = pd.read_csv(tune_df_path)

        test_df = None
        test_df_path = Path(fold_dir, "test.csv")
        if test_df_path.is_file():
            test_df = pd.read_csv(test_df_path)

        train_dataset_options = DatasetOptions(
            df=train_df,
            features_dir=features_dir,
            label_name=cfg.label_name,
            label_mapping=cfg.label_mapping,
        )
        tune_dataset_options = DatasetOptions(
            df=tune_df,
            features_dir=features_dir,
            label_name=cfg.label_name,
            label_mapping=cfg.label_mapping,
        )
        if test_df is not None:
            test_dataset_options = DatasetOptions(
                df=test_df,
                features_dir=features_dir,
                label_name=cfg.label_name,
                label_mapping=cfg.label_mapping,
            )

        print("Initializing datasets")
        train_dataset = ExtractedFeaturesDataset(train_dataset_options)
        tune_dataset = ExtractedFeaturesDataset(tune_dataset_options)
        if test_df is not None:
            test_dataset = ExtractedFeaturesDataset(test_dataset_options)

        m, n = train_dataset.num_classes, tune_dataset.num_classes
        assert (
            m == n == cfg.num_classes
        ), f"Either train (C={m}) or tune (C={n}) sets doesnt cover full class spectrum (C={cfg.num_classes})"

        criterion = LossFactory(cfg.task).get_loss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Initializing model")
        model = ModelFactory(
            level=cfg.model.level,
            num_classes=cfg.num_classes,
            options=cfg.model,
        ).get_model()
        model.to(device)
        print(model)

        print("Configuring optimizer & scheduler")
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

        print()
        with tqdm.tqdm(
            range(cfg.training.nepochs),
            desc=("Training"),
            unit=" epoch",
            leave=True,
        ) as t:
            for epoch in t:
                epoch_start_time = time.time()

                # set dataset seed
                train_dataset.seed = epoch

                if cfg.wandb.enable:
                    log_dict = {f"train/fold_{i}/epoch": epoch + 1}

                train_results = train(
                    epoch + 1,
                    model,
                    train_dataset,
                    optimizer,
                    criterion,
                    metric_names=cfg.metrics,
                    batch_size=cfg.training.batch_size,
                    gradient_accumulation=cfg.training.gradient_accumulation,
                    num_workers=num_workers,
                    device=device,
                )

                if cfg.wandb.enable:
                    update_log_dict(f"train/fold_{i}", train_results, log_dict, step=f"train/fold_{i}/epoch")

                train_dataset.df.to_csv(
                    Path(result_dir, f"train-{epoch+1}.csv"), index=False
                )

                if epoch % cfg.tuning.tune_every == 0:
                    tune_results = tune(
                        epoch + 1,
                        model,
                        tune_dataset,
                        criterion,
                        metric_names=cfg.metrics,
                        batch_size=cfg.tuning.batch_size,
                        num_workers=num_workers,
                        device=device,
                    )

                    if cfg.wandb.enable:
                        update_log_dict(f"tune/fold_{i}", tune_results, log_dict, step=f"train/fold_{i}/epoch")

                    tune_dataset.df.to_csv(
                        Path(result_dir, f"tune-{epoch+1}.csv"), index=False
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

                # logging
                if cfg.wandb.enable:
                    wandb.log(log_dict, step=epoch+1)

                epoch_end_time = time.time()
                epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
                tqdm.tqdm.write(
                    f"End of epoch {epoch+1} / {cfg.training.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
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
        best_model_fp = Path(checkpoint_dir, f"{cfg.testing.retrieve_checkpoint}.pt")
        if cfg.wandb.enable:
            wandb.save(str(best_model_fp))
        best_model_sd = torch.load(best_model_fp)
        model.load_state_dict(best_model_sd)

        # tune set inference
        best_tune_results = inference(
            model,
            tune_dataset,
            metric_names=cfg.metrics,
            batch_size=1,
            num_workers=num_workers,
            device=device,
        )
        tune_dataset.df.to_csv(
            Path(result_dir, f"tune-{cfg.testing.retrieve_checkpoint}.csv"), index=False
        )

        for r, v in best_tune_results.items():
            tune_metrics[f"fold_{i}"][r] = v
            if isinstance(v, float):
                v = round(v, 5)
            if cfg.wandb.enable:
                wandb.log({f"tune/fold_{i}/{r}-{cfg.testing.retrieve_checkpoint}": v})
            else:
                print(f"tune (fold {i}) {r}-{cfg.testing.retrieve_checkpoint}: {v}")

        if test_df is not None:
            # test set inference
            test_results = inference(
                model,
                test_dataset,
                metric_names=cfg.metrics,
                batch_size=1,
                num_workers=num_workers,
                device=device,
            )
            test_dataset.df.to_csv(Path(result_dir, "test.csv"), index=False)

            for r, v in test_results.items():
                test_metrics[f"fold_{i}"][r] = v
                if isinstance(v, float):
                    v = round(v, 5)
                if cfg.wandb.enable:
                    wandb.log({f"test/fold_{i}/{r}": v})
                else:
                    print(f"test (fold {i}) {r}: {v}")

        # freeing up memory at the end of each fold
        del model, train_dataset, tune_dataset, optimizer, scheduler, criterion
        if test_df is not None:
            del test_dataset

        # clear PyTorch's cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # explicitly calling garbage collection
        gc.collect()

    # gather mean metrics across folds
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
        metric_name: round(np.std(metric_values), 5)
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
        metric_name: round(np.std(metric_values), 5)
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
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)
