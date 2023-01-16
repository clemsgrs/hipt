import os
import time
import tqdm
import wandb
import hydra
import pandas as pd

from pathlib import Path
from omegaconf import DictConfig

from source.models import ModelFactory
from source.components import LossFactory
from source.dataset import ExtractedFeaturesSurvivalDataset
from source.utils import (
    initialize_wandb,
    train_survival,
    tune_survival,
    compute_time,
    log_on_step,
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

    print(f"Loading data")
    train_df = pd.read_csv(cfg.data.train_csv)
    tune_df = pd.read_csv(cfg.data.tune_csv)

    if cfg.training.pct:
        print(f"Training & Tuning on {cfg.training.pct*100}% of the data")
        train_df = train_df.sample(frac=cfg.training.pct).reset_index(drop=True)
        tune_df = tune_df.sample(frac=cfg.training.pct).reset_index(drop=True)

    train_dataset = ExtractedFeaturesSurvivalDataset(
        train_df, features_dir, cfg.label_name, nbins=cfg.nbins
    )
    tune_dataset = ExtractedFeaturesSurvivalDataset(
        tune_df, features_dir, cfg.label_name, nbins=cfg.nbins
    )

    # m, n = train_dataset.num_classes, tune_dataset.num_classes
    # assert (
    #     m == n == cfg.nbins
    # ), f"Either train (C={m}) or tune (C={n}) sets doesnt cover full class spectrum (C={cfg.nbins}"

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
        unit=" slide",
        ncols=100,
        leave=True,
    ) as t:

        for epoch in t:

            epoch_start_time = time.time()
            if cfg.wandb.enable:
                wandb.log({"epoch": epoch + 1})

            train_results = train_survival(
                epoch + 1,
                model,
                train_dataset,
                optimizer,
                criterion,
                batch_size=cfg.training.batch_size,
                gradient_accumulation=cfg.training.gradient_accumulation,
            )

            if cfg.wandb.enable:
                log_on_step("train", train_results, to_log=cfg.wandb.to_log)
            # train_dataset.df.to_csv(Path(result_dir, f"train_{epoch}.csv"), index=False)

            if epoch % cfg.tuning.tune_every == 0:

                tune_results = tune_survival(
                    epoch + 1,
                    model,
                    tune_dataset,
                    criterion,
                    batch_size=cfg.tuning.batch_size,
                )

                if cfg.wandb.enable:
                    log_on_step("tune", tune_results, to_log=cfg.wandb.to_log)
                # tune_dataset.df.to_csv(Path(result_dir, f"tune_{epoch}.csv"), index=False)

                early_stopping(epoch, model, tune_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            if cfg.wandb.enable:
                wandb.define_metric("train/lr", step_metric="epoch")
            if scheduler:
                lr = scheduler.get_last_lr()
                if cfg.wandb.enable:
                    wandb.log({"train/lr": lr})
                scheduler.step()
            elif cfg.wandb.enable:
                wandb.log({"train/lr": cfg.optim.lr})

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