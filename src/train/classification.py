import argparse
import os
import time
import tqdm
import wandb
import torch
import pandas as pd
import multiprocessing as mp

from pathlib import Path

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

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cfg.features_dir)

    num_workers = min(mp.cpu_count(), cfg.speed.num_workers)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    print("Loading data")
    train_df = pd.read_csv(cfg.data.train_csv)
    tune_df = pd.read_csv(cfg.data.tune_csv)
    if cfg.data.test_csv:
        test_df = pd.read_csv(cfg.data.test_csv)

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
    if cfg.data.test_csv:
        test_dataset_options = DatasetOptions(
            df=test_df,
            features_dir=features_dir,
            label_name=cfg.label_name,
            label_mapping=cfg.label_mapping,
        )

    print("Initializing datasets")
    train_dataset = ExtractedFeaturesDataset(train_dataset_options)
    tune_dataset = ExtractedFeaturesDataset(tune_dataset_options)
    if cfg.data.test_csv:
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
    start_time = time.time()

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
                log_dict = {"epoch": epoch + 1}

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
                update_log_dict("train", train_results, log_dict)

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
                    update_log_dict("tune", tune_results, log_dict)

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
                log_dict.update({"train/lr": lr})

            # logging
            if cfg.wandb.enable:
                wandb.log(log_dict, step=epoch + 1)

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
        if isinstance(v, float):
            v = round(v, 5)
        if cfg.wandb.enable:
            wandb.log({f"tune/{r}-{cfg.testing.retrieve_checkpoint}": v})
        else:
            print(f"tune {r}-{cfg.testing.retrieve_checkpoint}: {v}")

    if cfg.data.test_csv:
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
            if isinstance(v, float):
                v = round(v, 5)
            if cfg.wandb.enable:
                wandb.log({f"test/{r}": v})
            else:
                print(f"test {r}: {v}")

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f"Total time taken: {mins}m {secs}s")


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)
