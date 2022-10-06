import time
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import hydra
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from models import HIPT_4096, GlobalHIPT
from dataset import StackedTilesDataset
from utils import create_train_tune_test_df, train, tune, epoch_time

@hydra.main(version_base='1.2.0', config_path='config', config_name='default')
def main(cfg):

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # model = HIPT_4096(
    #     size_arg=cfg.size,
    #     dropout=cfg.dropout,
    #     num_classes=cfg.num_classes,
    #     pretrain_256=cfg.pretrain_256,
    #     freeze_256=cfg.freeze_256,
    #     pretrain_4096=cfg.pretrain_4096,
    #     freeze_4096=cfg.freeze_4096,
    # )

    model = GlobalHIPT(
        size_arg=cfg.size,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
    )

    if Path(cfg.data_dir, 'train.csv').exists() and Path(cfg.data_dir, 'tune.csv').exists() and Path(cfg.data_dir, 'test.csv').exists():
        train_df_path = Path(cfg.data_dir, 'train.csv')
        tune_df_path = Path(cfg.data_dir, 'tune.csv')
        test_df_path = Path(cfg.data_dir, 'test.csv')
        train_df = pd.read_csv(train_df_path)
        tune_df = pd.read_csv(tune_df_path)
        test_df = pd.read_csv(test_df_path)
    else:
        label_df = pd.read_csv(cfg.data_csv)
        train_df, tune_df, test_df = create_train_tune_test_df(label_df, save_csv=True, output_dir=cfg.data_dir)

    tiles_dir = Path(cfg.data_dir, 'patches')
    train_dataset = StackedTilesDataset(train_df, tiles_dir, tile_size=4096)
    tune_dataset = StackedTilesDataset(tune_df, tiles_dir, tile_size=4096)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()

    best_tune_loss = float('inf')
    if cfg.metric_objective == 'max':
        best_tune_metric = 0.0
    elif cfg.metric_objective == 'min':
        best_tune_metric = float('inf')

    train_losses, tune_losses = [], []
    train_metrics, tune_metrics = [], []

    for epoch in range(cfg.nepochs):

        start_time = time.time()

        train_loss, train_metric = train(
            epoch+1,
            model,
            train_dataset,
            optimizer,
            criterion,
            cfg.train_batch_size,
        )

        train_losses.append(train_loss)
        train_metrics.append(train_metric)

        if epoch % cfg.eval_every == 0:

            tune_loss, tune_metric = tune(
                epoch+1,
                model,
                tune_dataset,
                criterion,
                cfg.tune_batch_size
            )
            tune_losses.append(tune_loss)
            tune_metrics.append(tune_metric)

            if cfg.tracking == 'loss':
                if tune_loss < best_tune_loss:
                    best_tune_loss = tune_loss
                    model_fp = Path(output_dir, 'best_model.pt')
                    torch.save(model.state_dict(), model_fp)

            else:
                tune_metric_tracked = tune_metrics[cfg.tracking]
                if tune_metric_tracked > best_tune_metric:
                    best_tune_metric = tune_metric_tracked
                    model_fp = Path(output_dir, 'best_model.pt')
                    torch.save(model.state_dict(), model_fp)

        if cfg.lr_scheduler:
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'End of epoch {epoch+1} / {cfg.nepochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s')
        if cfg.tracking == 'loss':
            print(f'Train loss: {train_loss:.5f}')
            print(f'Tune loss: {tune_loss:.5f} (best Tune {cfg.tracking}: {best_tune_loss:.4f}\n')
        else:
            print(f'Train loss: {train_loss:.5f} \t Train {cfg.tracking}: {train_metrics[cfg.tracking]:.4f}')
            print(f'Tune loss: {tune_loss:.5f} \t Tune {cfg.tracking}: {tune_metrics[cfg.tracking]:.4f} (best Tune {cfg.tracking}: {best_tune_metric:.4f}\n')


if __name__ == '__main__':

    # python3 main.py
    main()



