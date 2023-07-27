import os
import time
import tqdm
import wandb
import torch
import hydra
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from omegaconf import OmegaConf, DictConfig

from source.models import ModelFactory
from source.components import LossFactory
from source.dataset import ClassificationDatasetOptions, DatasetFactory
from source.augmentations import AugmentationOptions, FeatureSpaceAugmentation
from source.utils import (
    initialize_wandb,
    train,
    train_ordinal,
    train_regression,
    tune,
    tune_ordinal,
    tune_regression,
    test,
    test_ordinal,
    test_regression,
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
    config_name="panda",
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

    features_root_dir = Path(cfg.features_root_dir)
    slide_features_dir = Path(features_root_dir, f"slide_features")
    region_features_dir = Path(features_root_dir, f"region_features")

    assert (cfg.task != "classification" and cfg.label_encoding != "ordinal") or (
        cfg.task == "classification"
    )

    print(f"Loading data")
    train_df = pd.read_csv(cfg.data.train_csv)
    tune_df = pd.read_csv(cfg.data.tune_csv)
    if cfg.data.test_csv:
        test_df = pd.read_csv(cfg.data.test_csv)

    if cfg.training.pct:
        print(f"Training on {cfg.training.pct*100}% of the data")
        train_df = train_df.sample(frac=cfg.training.pct).reset_index(drop=True)

    transform = None
    if cfg.augmentation.use:
        aug_dir = Path(output_dir, "augmentation")
        aug_dir.mkdir(parents=True, exist_ok=True)
        csv_path = Path(features_root_dir, "region_features.csv")
        if Path(csv_path).is_file():
            region_df = pd.read_csv(csv_path)
        elif cfg.augmentation.name in ["interpolation", "extrapolation"]:
            raise OSError(f"{csv_path} doesn't exist!")
        else:
            region_df = None
        kwargs = {k: v for e in cfg.augmentation.kwargs for k, v in e.items()}
        aug_options = AugmentationOptions(
            name=cfg.augmentation.name,
            output_dir=aug_dir,
            features_dir=region_features_dir,
            region_df=region_df,
            label_df=train_df,
            level=cfg.level,
            multiprocessing=(cfg.speed.num_workers == 0),
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
    )
    tune_dataset_options = ClassificationDatasetOptions(
        df=tune_df,
        features_dir=slide_features_dir,
        label_name=cfg.label_name,
        label_mapping=cfg.label_mapping,
        label_encoding=cfg.label_encoding,
    )
    if cfg.data.test_csv:
        test_dataset_options = ClassificationDatasetOptions(
            df=test_df,
            features_dir=slide_features_dir,
            label_name=cfg.label_name,
            label_mapping=cfg.label_mapping,
            label_encoding=cfg.label_encoding,
        )

    print(f"Initializing datasets")
    train_dataset = DatasetFactory(cfg.task, train_dataset_options).get_dataset()
    tune_dataset = DatasetFactory(cfg.task, tune_dataset_options).get_dataset()
    if cfg.data.test_csv:
        test_dataset = DatasetFactory(cfg.task, test_dataset_options).get_dataset()

    m, n = train_dataset.num_classes, tune_dataset.num_classes
    assert (
        m == n == cfg.num_classes
    ), f"Either train (C={m}) or tune (C={n}) sets doesnt cover full class spectrum (C={cfg.num_classes}"

    criterion = LossFactory(
        cfg.task, cfg.loss, cfg.label_encoding, cfg.loss_options
    ).get_loss()

    model = ModelFactory(
        cfg.level, cfg.num_classes, cfg.task, cfg.loss, cfg.label_encoding, cfg.model
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

            # set dataset seed
            train_dataset.seed = epoch

            if cfg.wandb.enable:
                log_dict = {"epoch": epoch + 1}

            if cfg.task == "regression":
                train_results = train_regression(
                    epoch + 1,
                    model,
                    train_dataset,
                    optimizer,
                    criterion,
                    batch_size=cfg.training.batch_size,
                    weighted_sampling=cfg.training.weighted_sampling,
                    gradient_accumulation=cfg.training.gradient_accumulation,
                    num_workers=cfg.speed.num_workers,
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
                    num_workers=cfg.speed.num_workers,
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
                    num_workers=cfg.speed.num_workers,
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
                if cfg.task == "regression":
                    tune_results = tune_regression(
                        epoch + 1,
                        model,
                        tune_dataset,
                        criterion,
                        batch_size=cfg.tuning.batch_size,
                        num_workers=cfg.speed.num_workers,
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
                        num_workers=cfg.speed.num_workers,
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
                        num_workers=cfg.speed.num_workers,
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

    if cfg.task == "regression":
        tune_results = test_regression(
            model,
            tune_dataset,
            batch_size=1,
            num_workers=cfg.speed.num_workers,
            use_wandb=cfg.wandb.enable,
        )
    elif cfg.label_encoding == "ordinal":
        tune_results = test_ordinal(
            model,
            tune_dataset,
            cfg.loss,
            batch_size=1,
            num_workers=cfg.speed.num_workers,
            use_wandb=cfg.wandb.enable,
        )
    else:
        tune_results = test(
            model,
            tune_dataset,
            collate_fn=partial(collate_features, label_type="int"),
            batch_size=1,
            num_workers=cfg.speed.num_workers,
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

    if cfg.data.test_csv:
        if cfg.task == "regression":
            test_results = test_regression(
                model,
                test_dataset,
                batch_size=1,
                num_workers=cfg.speed.num_workers,
                use_wandb=cfg.wandb.enable,
            )
        elif cfg.label_encoding == "ordinal":
            test_results = test_ordinal(
                model,
                test_dataset,
                cfg.loss,
                batch_size=1,
                num_workers=cfg.speed.num_workers,
                use_wandb=cfg.wandb.enable,
            )
        else:
            test_results = test(
                model,
                test_dataset,
                collate_fn=partial(collate_features, label_type="int"),
                batch_size=1,
                num_workers=cfg.speed.num_workers,
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
    main()
