import os
import time
import wandb
import torch
import torch.nn as nn
import hydra
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig

from source.models import ModelFactory
from source.dataset import StackedRegionsDataset
from source.utils import (
    initialize_wandb,
    train,
    tune,
    test,
    compute_time,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)


@hydra.main(version_base="1.2.0", config_path="../config/training/subtyping", config_name="region")
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir, cfg.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(output_dir, "checkpoints", cfg.level)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(output_dir, "results", cfg.level)
    result_dir.mkdir(parents=True, exist_ok=True)

    region_dir = Path(cfg.region_dir)

    # set up wandb
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        wandb_run.define_metric("lr", step_metric="epoch")
        print()

    model = ModelFactory(cfg.level, cfg.num_classes, cfg.model).get_model()
    model.relocate()
    print(model)

    print(f"Loading data")
    train_df = pd.read_csv(cfg.data.train_csv)
    tune_df = pd.read_csv(cfg.data.tune_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    if cfg.training.pct:
        print(f"Training & Tuning on {cfg.training.pct*100}% of the data")
        train_df = train_df.sample(frac=cfg.training.pct).reset_index(drop=True)
        tune_df = tune_df.sample(frac=cfg.training.pct).reset_index(drop=True)

    train_dataset = StackedRegionsDataset(
        train_df,
        region_dir,
        cfg.region_size,
        cfg.region_fmt,
        cfg.label_name,
        cfg.label_mapping,
        M_max=cfg.M_max,
    )
    tune_dataset = StackedRegionsDataset(
        tune_df,
        region_dir,
        cfg.region_size,
        cfg.region_fmt,
        cfg.label_name,
        cfg.label_mapping,
        M_max=cfg.M_max,
    )
    test_dataset = StackedRegionsDataset(
        test_df,
        region_dir,
        cfg.region_size,
        cfg.region_fmt,
        cfg.label_name,
        cfg.label_mapping,
        M_max=cfg.M_max,
    )

    m, n = train_dataset.num_classes, tune_dataset.num_classes
    assert (
        m == n == cfg.num_classes
    ), f"Either train (C={m}) or tune (C={n}) sets doesnt cover full class spectrum (C={cfg.num_classes}"

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
        save_all=cfg.early_stopping.save_all,
    )

    stop = False
    start_time = time.time()
    for epoch in range(cfg.nepochs):

        epoch_start_time = time.time()
        if cfg.wandb.enable:
            wandb.log({"epoch": epoch + 1})

        train_results = train(
            epoch + 1,
            model,
            train_dataset,
            optimizer,
            criterion,
            batch_size=cfg.training.batch_size,
            weighted_sampling=cfg.training.weighted_sampling,
            gradient_accumulation=cfg.training.gradient_accumulation,
        )

        train_dataset.df.to_csv(Path(result_dir, f"train_{epoch}.csv"), index=False)
        if cfg.wandb.enable:
            for res, val in train_results.items():
                wandb.define_metric(f"train/{res}", step_metric="epoch")
                wandb.log({f"train/{res}": val})

        if epoch % cfg.tuning.tune_every == 0:

            tune_results = tune(
                epoch + 1,
                model,
                tune_dataset,
                criterion,
                batch_size=cfg.tuning.batch_size,
            )

            tune_dataset.df.to_csv(Path(result_dir, f"tune_{epoch}.csv"), index=False)
            if cfg.wandb.enable:
                for res, val in tune_results.items():
                    wandb.define_metric(f"tune/{res}", step_metric="epoch")
                    wandb.log({f"tune/{res}": val})

            early_stopping(epoch, model, tune_results)
            if early_stopping.early_stop and cfg.early_stopping.enable:
                stop = True

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

    # load best model
    best_model_sd = torch.load(Path(checkpoint_dir, f"{cfg.testing.retrieve_checkpoint}_model.pt"))
    model.load_state_dict(best_model_sd)

    test_results = test(model, test_dataset, batch_size=1)
    test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)

    test_auc = round(test_results["auc"], 2)
    if cfg.wandb.enable:
        wandb.log({f"test/auc": test_auc})

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    main()
