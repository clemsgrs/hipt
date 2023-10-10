import os
import time
import tqdm
import wandb
import torch
import hydra
import datetime
import statistics
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig

from source.models import ModelFactory
from source.components import LossFactory
from source.dataset import (
    SurvivalDatasetOptions,
    DatasetFactory,
    ppcess_survival_data,
    ppcess_tcga_survival_data,
)
from source.utils import (
    initialize_wandb,
    train_survival,
    tune_survival,
    test_survival,
    compute_time,
    update_log_dict,
    aggregated_cindex,
    get_cumulative_dynamic_auc,
    plot_cumulative_dynamic_auc,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)


@hydra.main(
    version_base="1.2.0", config_path="../config/training/survival", config_name="multi"
)
def main(cfg: DictConfig):
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        log_to_wandb = {k: v for e in cfg.wandb.to_log for k, v in e.items()}
        run_id = wandb_run.id

    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_root_dir = Path(output_dir, "checkpoints")
    checkpoint_root_dir.mkdir(parents=True, exist_ok=True)

    result_root_dir = Path(output_dir, "results")
    result_root_dir.mkdir(parents=True, exist_ok=True)

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

    fold_root_dir = Path(cfg.data.fold_dir)
    nfold = len([_ for _ in fold_root_dir.glob(f"fold_*")])
    print(f"Training on {nfold} folds")

    test_metrics = []

    start_time = time.time()
    for i in range(nfold):
        fold_dir = Path(fold_root_dir, f"fold_{i}")
        checkpoint_dir = Path(checkpoint_root_dir, f"fold_{i}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir = Path(result_root_dir, f"fold_{i}")
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading data for fold {i}")
        dfs = {}
        for p in ["train", "tune", "test"]:
            df_path = Path(fold_dir, f"{p}.csv")
            df = pd.read_csv(df_path)
            df["partition"] = [p] * len(df)
            dfs[p] = df

        if cfg.training.pct:
            print(f"Training on {cfg.training.pct*100}% of the data")
            dfs["train"] = (
                dfs["train"].sample(frac=cfg.training.pct).reset_index(drop=True)
            )

        df = pd.concat([df for df in dfs.values()], ignore_index=True)
        patient_df, slide_df = ppcess_survival_data(df, cfg.label_name, nbins=cfg.nbins)

        patient_dfs, slide_dfs = {}, {}
        for p in ["train", "tune", "test"]:
            patient_dfs[p] = patient_df[patient_df.partition == p].reset_index(
                drop=True
            )
            slide_dfs[p] = slide_df[slide_df.partition == p]

        train_dataset_options = SurvivalDatasetOptions(
            patient_df=patient_dfs["train"],
            slide_df=slide_dfs["train"],
            tiles_df=tiles_df,
            features_dir=features_dir,
            label_name=cfg.label_name,
        )
        tune_dataset_options = SurvivalDatasetOptions(
            patient_df=patient_dfs["tune"],
            slide_df=slide_dfs["tune"],
            tiles_df=tiles_df,
            features_dir=features_dir,
            label_name=cfg.label_name,
        )
        test_dataset_options = SurvivalDatasetOptions(
            patient_df=patient_dfs["test"],
            slide_df=slide_dfs["test"],
            tiles_df=tiles_df,
            features_dir=features_dir,
            label_name=cfg.label_name,
        )

        train_dataset = DatasetFactory(
            "survival", train_dataset_options, cfg.model.agg_method
        ).get_dataset()
        tune_dataset = DatasetFactory(
            "survival", tune_dataset_options, cfg.model.agg_method
        ).get_dataset()
        test_dataset = DatasetFactory(
            "survival", test_dataset_options, cfg.model.agg_method
        ).get_dataset()

        model = ModelFactory(
            cfg.level, cfg.nbins, "survival", cfg.loss, cfg.label_encoding, cfg.model
        ).get_model()
        model.relocate()
        print(model)

        print("Configuring optimmizer & scheduler")
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = OptimizerFactory(
            cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
        ).get_optimizer()
        scheduler = SchedulerFactory(optimizer, cfg.optim.lr_scheduler).get_scheduler()

        criterion = LossFactory(cfg.task, cfg.loss).get_loss()

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
            desc=(f"Fold {i} Training"),
            unit=" epoch",
            ncols=100,
            leave=True,
        ) as t:
            for epoch in t:

                # set dataset seed
                train_dataset.seed = epoch
                tune_dataset.seed = epoch

                epoch_start_time = time.time()
                if cfg.wandb.enable:
                    log_dict = {f"train/fold_{i}/epoch": epoch + 1}

                train_results = train_survival(
                    epoch + 1,
                    model,
                    train_dataset,
                    optimizer,
                    criterion,
                    agg_method=cfg.model.agg_method,
                    batch_size=cfg.training.batch_size,
                    weighted_sampling=cfg.training.weighted_sampling,
                    gradient_accumulation=cfg.training.gradient_accumulation,
                    num_workers=num_workers,
                )

                if cfg.wandb.enable:
                    update_log_dict(
                        f"train/fold_{i}",
                        train_results,
                        log_dict,
                        step=f"train/fold_{i}/epoch",
                        to_log=log_to_wandb["train"],
                    )

                train_dataset.df.to_csv(
                    Path(result_dir, f"train_{epoch+1}.csv"), index=False
                )

                if epoch % cfg.tuning.tune_every == 0:
                    tune_results = tune_survival(
                        epoch + 1,
                        model,
                        tune_dataset,
                        criterion,
                        agg_method=cfg.model.agg_method,
                        batch_size=cfg.tuning.batch_size,
                        num_workers=num_workers,
                    )

                    auc, mean_auc, times = None, None, None
                    # auc, mean_auc, times = get_cumulative_dynamic_auc(
                    #     patient_dfs["train"],
                    #     patient_dfs["tune"],
                    #     tune_results["risks"],
                    #     cfg.label_name,
                    # )
                    if cfg.wandb.enable:
                        update_log_dict(
                            f"tune/fold_{i}",
                            tune_results,
                            log_dict,
                            step=f"train/fold_{i}/epoch",
                            to_log=log_to_wandb["tune"],
                        )
                        if auc is not None:
                            fig = plot_cumulative_dynamic_auc(
                                auc, mean_auc, times, epoch
                            )
                            log_dict.update(
                                {
                                    f"tune/fold_{i}/cumulative_dynamic_auc": wandb.Image(
                                        fig
                                    )
                                }
                            )
                            plt.close(fig)

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

                # logging
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
        print(f"Total time taken for fold {i}: {fold_mins}m {fold_secs}s")

        # load best model
        best_model_fp = Path(checkpoint_dir, f"{cfg.testing.retrieve_checkpoint}.pt")
        if cfg.wandb.enable:
            wandb.save(str(best_model_fp))
        best_model_sd = torch.load(best_model_fp)
        model.load_state_dict(best_model_sd)

        test_results = test_survival(
            model,
            test_dataset,
            agg_method=cfg.model.agg_method,
            batch_size=1,
            num_workers=num_workers,
        )
        test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)

        for r, v in test_results.items():
            if r == "c-index":
                if test_dataset.agg_level == "slide":
                    v = aggregated_cindex(test_dataset.df, label_name=cfg.label_name)
                    test_metrics.append(v)
                    v = round(v, 5)
                else:
                    test_metrics.append(v)
                    v = round(v, 5)
            if cfg.wandb.enable and r in log_to_wandb["test"]:
                wandb.log({f"test/fold_{i}/{r}": v})
            elif "cm" not in r:
                print(f"test {r}: {v}")

    mean_test_metric = round(np.mean(test_metrics), 5)
    std_test_metric = round(statistics.stdev(test_metrics), 5)
    if cfg.wandb.enable and "c-index" in log_to_wandb["test"]:
        wandb.log({f"test/c-index_mean": mean_test_metric})
        wandb.log({f"test/c-index_std": std_test_metric})

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken ({nfold} folds): {mins}m {secs}s")


if __name__ == "__main__":
    main()
