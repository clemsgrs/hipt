import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import hydra
import datetime
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig

from source.models import ModelFactory
from source.components import LossFactory
from source.dataset import StackedRegionsDataset
from source.utils import (
    initialize_wandb,
    train_on_regions,
    tune_on_regions,
    test_on_regions,
    compute_time,
    update_log_dict,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
)


@hydra.main(
    version_base="1.2.0",
    config_path="../config/training/classification",
    config_name="region",
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

    checkpoint_dir = Path(output_dir, "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(output_dir, "results")
    result_dir.mkdir(parents=True, exist_ok=True)

    region_dir = Path(cfg.region_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

    assert (cfg.task != "classification" and cfg.label_encoding != "ordinal") or (
        cfg.task == "classification"
    )
    criterion = LossFactory(
        cfg.task, cfg.loss, cfg.label_encoding, cfg.loss_options
    ).get_loss()

    model = ModelFactory(
        cfg.level, cfg.num_classes, cfg.task, cfg.loss, cfg.label_encoding, cfg.model
    ).get_model()
    model.relocate()
    print(model)

    print(f"Loading data")
    train_df = pd.read_csv(cfg.data.train_csv)
    tune_df = pd.read_csv(cfg.data.tune_csv)
    if cfg.data.test_csv:
        test_df = pd.read_csv(cfg.data.test_csv)

    if cfg.training.pct:
        print(f"Training & Tuning on {cfg.training.pct*100}% of the data")
        train_df = train_df.sample(frac=cfg.training.pct).reset_index(drop=True)
        tune_df = tune_df.sample(frac=cfg.training.pct).reset_index(drop=True)

    train_dataset = StackedRegionsDataset(
        train_df,
        region_dir,
        cfg.model.region_size,
        cfg.region_fmt,
        cfg.label_name,
        cfg.label_mapping,
        M_max=cfg.M_max,
    )
    tune_dataset = StackedRegionsDataset(
        tune_df,
        region_dir,
        cfg.model.region_size,
        cfg.region_fmt,
        cfg.label_name,
        cfg.label_mapping,
        M_max=cfg.M_max,
    )
    if cfg.data.test_csv:
        test_dataset = StackedRegionsDataset(
            test_df,
            region_dir,
            cfg.model.region_size,
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

    with tqdm.tqdm(
        range(cfg.nepochs),
        desc=(f"HIPT Training"),
        unit=" epoch",
        ncols=100,
        leave=True,
    ) as t:
        for epoch in t:
            epoch_start_time = time.time()

            # set dataset seed
            train_dataset.seed = epoch
            tune_dataset.seed = epoch

            if cfg.wandb.enable:
                log_dict = {"epoch": epoch + 1}

            train_results = train_on_regions(
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

            if cfg.wandb.enable:
                update_log_dict(
                    "train", train_results, log_dict, to_log=log_to_wandb["train"]
                )
            train_dataset.df.to_csv(
                Path(result_dir, f"train_{epoch+1}.csv"), index=False
            )

            if epoch % cfg.tuning.tune_every == 0:
                tune_results = tune_on_regions(
                    epoch + 1,
                    model,
                    tune_dataset,
                    criterion,
                    batch_size=cfg.tuning.batch_size,
                    num_workers=num_workers,
                    use_wandb=cfg.wandb.enable,
                )

                if cfg.wandb.enable:
                    update_log_dict(
                        "tune",
                        tune_results,
                        log_dict,
                        to_log=[e for e in log_to_wandb["tune"] if "cm" not in e],
                    )
                tune_dataset.df.to_csv(
                    Path(result_dir, f"tune_{epoch+1}.csv"), index=False
                )

                early_stopping(epoch, model, tune_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            lr = cfg.optim.lr
            if scheduler:
                lr = scheduler.get_last_lr()
                scheduler.step()
            if cfg.wandb.enable:
                log_dict.update({"train/lr": lr})

            # logging
            if cfg.wandb.enable:
                wandb.log(log_dict, step=epoch + 1)

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
    best_model_fp = Path(checkpoint_dir, f"{cfg.testing.retrieve_checkpoint}.pt")
    if cfg.wandb.enable:
        wandb.save(str(best_model_fp))
    best_model_sd = torch.load(best_model_fp)
    model.load_state_dict(best_model_sd)

    # best tune score
    tune_dataset.seed = early_stopping.best_epoch
    tune_results = test_on_regions(
        model,
        tune_dataset,
        batch_size=1,
        num_workers=num_workers,
        use_wandb=cfg.wandb.enable,
    )
    tune_dataset.df.to_csv(
        Path(result_dir, f"tune_{cfg.testing.retrieve_checkpoint}.csv"), index=False
    )

    for r, v in tune_results.items():
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
                        f"tune/{r}_{cfg.testing.retrieve_checkpoint}": wandb.Image(
                            str(save_path)
                        )
                    }
                )
            else:
                wandb.log({f"tune/{r}_{cfg.testing.retrieve_checkpoint}": v})
        elif "cm" not in r:
            print(f"tune {r}_{cfg.testing.retrieve_checkpoint}: {v}")

    # testing
    if cfg.data.test_csv:
        test_dataset.seed = early_stopping.best_epoch
        test_results = test_on_regions(
            model,
            test_dataset,
            batch_size=1,
            num_workers=num_workers,
            use_wandb=cfg.wandb.enable,
        )
        test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)

        for r, v in test_results.items():
            if isinstance(v, float):
                v = round(v, 5)
            if r == "cm":
                save_path = Path(result_dir, f"test_cm.png")
                v.savefig(save_path, bbox_inches="tight")
                plt.close(v)
            if cfg.wandb.enable and r in log_to_wandb["test"]:
                if r == "cm":
                    wandb.log({f"test/{r}": wandb.Image(str(save_path))})
                else:
                    wandb.log({f"test/{r}": v})
            elif "cm" not in r:
                print(f"test {r}: {v}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy("file_system")

    main()
