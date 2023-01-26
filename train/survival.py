import os
import time
import tqdm
import wandb
import hydra
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig

from source.models import ModelFactory
from source.components import LossFactory
from source.dataset import ExtractedFeaturesSurvivalDataset, ppcess_tcga_survival_data
from source.utils import (
    initialize_wandb,
    train_survival,
    tune_survival,
    compute_time,
    update_log_dict,
    get_cumulative_dynamic_auc,
    plot_cumulative_dynamic_auc,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)


@hydra.main(version_base="1.2.0", config_path="../config/training/survival", config_name="debug")
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir, cfg.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(output_dir, "checkpoints", cfg.level)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(output_dir, "results", cfg.level)
    result_dir.mkdir(parents=True, exist_ok=True)

    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")

    features_dir = Path(output_dir, "features", cfg.level)
    if cfg.features_dir:
        features_dir = Path(cfg.features_dir)

    criterion = LossFactory(cfg.task, cfg.loss).get_loss()

    model = ModelFactory(cfg.level, num_classes=cfg.nbins, model_options=cfg.model).get_model()
    model.relocate()
    print(model)

    print("Loading data")
    train_df = pd.read_csv(cfg.data.train_csv)
    tune_df = pd.read_csv(cfg.data.tune_csv)

    if cfg.training.pct:
        print(f"Training & Tuning on {cfg.training.pct*100}% of the data")
        train_df = train_df.sample(frac=cfg.training.pct).reset_index(drop=True)
        tune_df = tune_df.sample(frac=cfg.training.pct).reset_index(drop=True)

    train_df['partition'] = ['train'] * len(train_df)
    tune_df['partition'] = ['tune'] * len(tune_df)

    train_tune_df = pd.concat([train_df, tune_df], ignore_index=True)
    patient_df, slide_df = ppcess_tcga_survival_data(train_tune_df, cfg.label_name, nbins=cfg.nbins)

    train_patient_df = patient_df[patient_df.partition == 'train'].reset_index(drop=True)
    tune_patient_df = patient_df[patient_df.partition == 'tune'].reset_index(drop=True)
    train_slide_df = slide_df[slide_df.partition == 'train']
    tune_slide_df = slide_df[slide_df.partition == 'tune']

    train_dataset = ExtractedFeaturesSurvivalDataset(
        train_patient_df, train_slide_df, features_dir, cfg.label_name,
    )
    tune_dataset = ExtractedFeaturesSurvivalDataset(
        tune_patient_df, tune_slide_df, features_dir, cfg.label_name,
    )

    print("Configuring optimmizer & scheduler")
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
    start_time = time.time()

    with tqdm.tqdm(
        range(cfg.nepochs),
        desc=(f"HIPT Training"),
        unit=" patient",
        ncols=100,
        leave=True,
    ) as t:

        for epoch in t:

            epoch_start_time = time.time()
            if cfg.wandb.enable:
                log_dict = {"epoch": epoch+1}

            train_results = train_survival(
                epoch+1,
                model,
                train_dataset,
                optimizer,
                criterion,
                batch_size=cfg.training.batch_size,
                gradient_accumulation=cfg.training.gradient_accumulation,
            )

            if cfg.wandb.enable:
                update_log_dict("train", train_results, log_dict, to_log=cfg.wandb.to_log)
            # train_dataset.df.to_csv(Path(result_dir, f"train_{epoch}.csv"), index=False)

            if epoch % cfg.tuning.tune_every == 0:

                tune_results = tune_survival(
                    epoch+1,
                    model,
                    tune_dataset,
                    criterion,
                    batch_size=cfg.tuning.batch_size,
                )

                auc, mean_auc, times = get_cumulative_dynamic_auc(train_patient_df, tune_patient_df, tune_results["risks"], cfg.label_name)
                if cfg.wandb.enable:
                    update_log_dict("tune", tune_results, log_dict, to_log=cfg.wandb.to_log)
                    fig = plot_cumulative_dynamic_auc(auc, mean_auc, times, epoch)
                    log_dict.update({"tune/cumulative_dynamic_auc": wandb.Image(fig)})
                    plt.close(fig)
                # tune_dataset.df.to_csv(Path(result_dir, f"tune_{epoch}.csv"), index=False)

                early_stopping(epoch, model, tune_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            lr = cfg.optim.lr
            if scheduler:
                lr = scheduler.get_last_lr()[0]
                scheduler.step()
            if cfg.wandb.enable:
                log_dict.update({"train/lr": lr})

            # logging
            if cfg.wandb.enable:
                wandb.log(log_dict, step=epoch+1)

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

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    main()
