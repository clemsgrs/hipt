import os
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from source.models import ModelFactory
from source.dataset import ExtractedFeaturesDataset
from source.utils import initialize_wandb, train, tune, test, compute_time, EarlyStopping, OptimizerFactory, SchedulerFactory


@hydra.main(version_base='1.2.0', config_path='config', config_name='multifold_global')
def main(cfg):

    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(output_dir, 'checkpoints', cfg.level)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_root_dir = Path(output_dir, 'results', cfg.level)
    result_root_dir.mkdir(parents=True, exist_ok=True)

    # set up wandb
    key = os.environ.get('WANDB_API_KEY')
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _ = initialize_wandb(project=cfg.wandb.project, exp_name=cfg.wandb.exp_name, entity=cfg.wandb.username, config=config, key=key)
    wandb.define_metric('epoch', summary='max')
    wandb.define_metric('lr', step_metric='epoch')

    if cfg.features_dir:
        features_dir = Path(cfg.features_dir)
    else:
        features_dir = Path(output_dir, 'features', cfg.level)

    fold_root_dir = Path(cfg.data_dir, cfg.dataset_name, 'splits')
    nfold = len([_ for _ in fold_root_dir.glob(f'fold_*')])

    test_aucs = []

    start_time = time.time()
    for i in range(nfold):

        fold_dir = Path(fold_root_dir, f'fold_{i}')
        result_dir = Path(result_root_dir, f'fold_{i}')
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f'Loading data for fold {i}')
        train_df_path = Path(fold_dir, 'train.csv')
        tune_df_path = Path(fold_dir, 'tune.csv')
        test_df_path = Path(fold_dir, 'test.csv')
        train_df = pd.read_csv(train_df_path)
        tune_df = pd.read_csv(tune_df_path)
        test_df = pd.read_csv(test_df_path)

        if cfg.pct:
            print(f'Training on {cfg.pct*100}% of the data')
            train_df = train_df.sample(frac=cfg.pct).reset_index(drop=True)

        train_dataset = ExtractedFeaturesDataset(train_df, features_dir, cfg.label_name, cfg.label_mapping)
        tune_dataset = ExtractedFeaturesDataset(tune_df, features_dir, cfg.label_name, cfg.label_mapping)
        test_dataset = ExtractedFeaturesDataset(test_df, features_dir, cfg.label_name, cfg.label_mapping)

        train_c, tune_c, test_c = train_dataset.num_classes, tune_dataset.num_classes, test_dataset.num_classes
        assert train_c == tune_c == test_c, f'Different number of classes C in train (C={train_c}), tune (C={tune_c}) and test (C={test_c}) sets!'

        model = ModelFactory(cfg.level, cfg.num_classes, cfg.model).get_model()
        model.relocate()
        print(model)

        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = OptimizerFactory(cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd).get_optimizer()
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
        for epoch in range(cfg.nepochs):

            epoch_start_time = time.time()
            wandb.log({'epoch': epoch+1})

            train_results = train(
                epoch+1,
                model,
                train_dataset,
                optimizer,
                criterion,
                batch_size=cfg.train_batch_size,
                weighted_sampling=cfg.weighted_sampling,
                gradient_clipping=cfg.gradient_clipping,
            )

            train_dataset.df.to_csv(Path(result_dir, f'train_{epoch}.csv'), index=False)
            for res, val in train_results.items():
                wandb.define_metric(f'train/fold_{i}/{res}', step_metric='epoch')
                wandb.log({f'train/fold_{i}/{res}': val})

            if epoch % cfg.tune_every == 0:

                tune_results = tune(
                    epoch+1,
                    model,
                    tune_dataset,
                    criterion,
                    batch_size=cfg.tune_batch_size
                )

                tune_dataset.df.to_csv(Path(result_dir, f'tune_{epoch}.csv'), index=False)
                for res, val in tune_results.items():
                    wandb.define_metric(f'tune/fold_{i}/{res}', step_metric='epoch')
                    wandb.log({f'tune/fold_{i}/{res}': val})

                early_stopping(epoch, model, tune_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    stop = True

            if scheduler:
                lr = scheduler.get_last_lr()
                wandb.log({f'train/fold_{i}/lr': lr})
                scheduler.step()
            else:
                wandb.log({f'train/fold_{i}/lr': cfg.optim.lr})

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            print(f'End of epoch {epoch+1} / {cfg.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s')

            if stop:
                print(f'Stopping early because best {cfg.early_stopping.tracking} was reached {cfg.early_stopping.patience} epochs ago')
                break

        fold_end_time = time.time()
        fold_mins, fold_secs = compute_time(fold_start_time, fold_end_time)
        print(f'Total time taken for fold {i}: {fold_mins}m {fold_secs}s')

        # load best model
        best_model_fp = Path(checkpoint_dir, f'best_model_{wandb.run.id}.pt')
        wandb.save(best_model_fp)
        best_model_sd = torch.load(best_model_fp)
        model.load_state_dict(best_model_sd)

        test_results = test(model, test_dataset, batch_size=1)
        test_dataset.df.to_csv(Path(result_dir, f'test.csv'), index=False)

        test_auc = round(test_results['auc'], 2)
        wandb.log({f'test/fold_{i}/auc': test_auc})
        test_aucs.append(test_auc)

    mean_test_auc = round(np.mean(test_aucs), 2)
    wandb.log({f'test/mean_auc': mean_test_auc})

    end_time = time.time()
    mins, secs = compute_time(start_time, end_time)
    print(f'Total time taken ({nfold} folds): {mins}m {secs}s')

if __name__ == '__main__':

    # python3 train_global.py
    main()



