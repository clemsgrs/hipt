import os
import time
import wandb
import torch
import torch.nn as nn
import hydra
import statistics
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from source.models import ModelFactory
from source.dataset import ExtractedFeaturesDataset
from source.utils import (
    initialize_wandb,
    train,
    tune,
    test,
    compute_time,
    log_on_step,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)


@hydra.main(
    version_base="1.2.0", config_path="config/training", config_name="multifold_global"
)
def main(cfg):

    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_root_dir = Path(output_dir, "checkpoints", cfg.level)
    checkpoint_root_dir.mkdir(parents=True, exist_ok=True)

    result_root_dir = Path(output_dir, "results", cfg.level)
    result_root_dir.mkdir(parents=True, exist_ok=True)

    # set up wandb
    key = os.environ.get("WANDB_API_KEY")
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _ = initialize_wandb(
        cfg.wandb.project,
        cfg.wandb.username,
        cfg.wandb.exp_name,
        dir=cfg.wandb.dir,
        config=config,
        key=key,
    )

    if cfg.features_dir:
        features_dir = Path(cfg.features_dir)
    else:
        features_dir = Path(output_dir, "features", cfg.level)

    fold_root_dir = Path(cfg.data_dir, cfg.dataset_name, "splits")
    nfold = len([_ for _ in fold_root_dir.glob(f"fold_*")])
    print(f"Training on {nfold} folds")

    test_aucs = []

    start_time = time.time()
    for i in range(nfold):

        fold_dir = Path(fold_root_dir, f"fold_{i}")
        checkpoint_dir = Path(checkpoint_root_dir, f"fold_{i}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir = Path(result_root_dir, f"fold_{i}")
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading data for fold {i}")
        train_df_path = Path(fold_dir, "train.csv")
        tune_df_path = Path(fold_dir, "tune.csv")
        test_df_path = Path(fold_dir, "test.csv")
        train_df = pd.read_csv(train_df_path)
        tune_df = pd.read_csv(tune_df_path)
        test_df = pd.read_csv(test_df_path)

        if cfg.pct:
            print(f"Training on {cfg.pct*100}% of the data")
            train_df = train_df.sample(frac=cfg.pct).reset_index(drop=True)

        train_dataset = ExtractedFeaturesDataset(
            train_df, features_dir, cfg.label_name, cfg.label_mapping
        )
        tune_dataset = ExtractedFeaturesDataset(
            tune_df, features_dir, cfg.label_name, cfg.label_mapping
        )
        test_dataset = ExtractedFeaturesDataset(
            test_df, features_dir, cfg.label_name, cfg.label_mapping
        )

        train_c, tune_c, test_c = (
            train_dataset.num_classes,
            tune_dataset.num_classes,
            test_dataset.num_classes,
        )
        assert (
            train_c == tune_c == test_c
        ), f"Different number of classes C in train (C={train_c}), tune (C={tune_c}) and test (C={test_c}) sets!"

        model = ModelFactory(cfg.level, cfg.num_classes, cfg.model).get_model()
        model.relocate()
        print(model)

        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = OptimizerFactory(
            cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
        ).get_optimizer()
        scheduler = SchedulerFactory(optimizer, cfg.optim.lr_scheduler).get_scheduler()

        criterion = nn.CrossEntropyLoss()

        early_stopping = EarlyStopping(
            cfg.early_stopping.tracking,
            cfg.early_stopping.min_max,
            cfg.early_stopping.patience,
            cfg.early_stopping.min_epoch,
            checkpoint_dir=checkpoint_dir,
            save_all=cfg.save_all,
        )

        stop = False
        fold_start_time = time.time()

        wandb.define_metric(f"fold_{i}/train/epoch", summary="max")
        for epoch in range(cfg.nepochs):

            epoch_start_time = time.time()
            wandb.log({f"fold_{i}/train/epoch": epoch})

            train_results = train(
                epoch + 1,
                model,
                train_dataset,
                optimizer,
                criterion,
                batch_size=cfg.train_batch_size,
                weighted_sampling=cfg.weighted_sampling,
                gradient_clipping=cfg.gradient_clipping,
            )

            log_on_step(
                f"fold_{i}/train",
                train_results,
                step=f"fold_{i}/train/epoch",
                to_log=cfg.wandb.to_log,
            )
            train_dataset.df.to_csv(Path(result_dir, f"train_{epoch}.csv"), index=False)

            if epoch % cfg.tune_every == 0:

                tune_results = tune(
                    epoch + 1,
                    model,
                    tune_dataset,
                    criterion,
                    batch_size=cfg.tune_batch_size,
                )

                log_on_step(
                    f"fold_{i}/tune",
                    tune_results,
                    step=f"fold_{i}/train/epoch",
                    to_log=cfg.wandb.to_log,
                )
                tune_dataset.df.to_csv(
                    Path(result_dir, f"tune_{epoch}.csv"), index=False
                )

                early_stopping(epoch, model, tune_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            wandb.define_metric(
                f"fold_{i}/train/lr", step_metric=f"fold_{i}/train/epoch"
            )
            if scheduler:
                lr = scheduler.get_last_lr()
                wandb.log({f"fold_{i}/train/lr": lr})
                scheduler.step()
            else:
                wandb.log({f"fold_{i}/train/lr": cfg.optim.lr})

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            print(
                f"End of epoch {epoch+1} / {cfg.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
            )

            if stop:
                print(
                    f"Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago"
                )
                break

        fold_end_time = time.time()
        fold_mins, fold_secs = compute_time(fold_start_time, fold_end_time)
        print(f"Total time taken for fold {i}: {fold_mins}m {fold_secs}s")

        # load best model
        best_model_fp = Path(checkpoint_dir, f"best_model_{wandb.run.id}.pt")
        wandb.save(str(best_model_fp))
        best_model_sd = torch.load(best_model_fp)
        model.load_state_dict(best_model_sd)

        test_results = test(model, test_dataset, batch_size=1)
        test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)

        for r, v in test_results.items():
            if r == "auc":
                test_aucs.append(v)
                v = round(v, 3)
            if r in cfg.wandb.to_log:
                wandb.log({f"fold_{i}/test/{r}": v})

    mean_test_auc = round(np.mean(test_aucs), 3)
    std_test_auc = round(statistics.stdev(test_aucs), 3)
    wandb.log({f"test/auc_mean": mean_test_auc})
    wandb.log({f"test/auc_std": std_test_auc})

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken ({nfold} folds): {mins}m {secs}s")


if __name__ == "__main__":

    # python3 train_global.py
    main()
