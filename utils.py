import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from typing import Optional, Callable
from sklearn import metrics
from sklearn.model_selection import train_test_split


def extract_coord_from_path(path):
    '''
    Path expected to look like /path/to/dir/x_y.png
    '''
    x_y = path.stem
    x, y = x_y.split('_')[0], x_y.split('_')[1]
    return int(x), int(y)


def update_state_dict(model_dict, state_dict):
    success, failure = 0, 0
    updated_state_dict = {}
    for k,v in zip(model_dict.keys(), state_dict.values()):
        if v.size() != model_dict[k].size():
            updated_state_dict[k] = model_dict[k]
            failure += 1
        else:
            updated_state_dict[k] = v
            success += 1
    msg = f'{success} weight(s) loaded succesfully ; {failure} weight(s) not loaded because of mismatching shapes'
    return updated_state_dict, msg


def create_train_tune_test_df(
    df: pd.DataFrame,
    save_csv: bool = False,
    output_dir: Path = Path(''),
    tune_size: float = .4,
    test_size: float = .2,
    seed: Optional[int] = 21,
    ):
    train_df, tune_df = train_test_split(df, test_size=tune_size, random_state=seed)
    train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=seed)
    if save_csv:
        train_df.to_csv(Path(output_dir, f'train.csv'), index=False)
        tune_df.to_csv(Path(output_dir, f'tune.csv'), index=False)
        test_df.to_csv(Path(output_dir, f'test.csv'), index=False)
    train_df = train_df.reset_index()
    tune_df = tune_df.reset_index()
    test_df = test_df.reset_index()
    return train_df, tune_df, test_df


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_metrics(probs, labels, threshold: float = 0.5):
    probs, labels = np.asarray(probs), np.asarray(labels)
    preds = probs > threshold
    acc = metrics.accuracy_score(labels, preds)
    auc = metrics.roc_auc_score(labels, probs)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds)
    
    metrics_dict = {'acc': acc, 'auc': auc, 'precision': precision, 'recall': recall}
    return metrics_dict


def train(
    epoch: int,
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    batch_size: Optional[int] = 1,
    threshold: Optional[float] = 0.5,
    ):

    model.train()
    epoch_loss = 0
    probs = []
    labels = []
    idxs = []

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    with tqdm.tqdm(
        train_loader,
        desc=(f'Train - Epoch {epoch}'),
        unit=' slide',
        ncols=80,
        unit_scale=batch_size,
        position=0,
        leave=True) as t:

        for i, batch in enumerate(t):

            optimizer.zero_grad()
            idx, stacked_tiles, label = batch
            stacked_tiles, label = stacked_tiles.cuda(device='cuda:1'), label.cuda(device='cuda:0')
            logits = model(stacked_tiles)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            prob = torch.sigmoid(logits)
            probs.extend(prob[:,0].clone().tolist())
            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

            epoch_loss += loss.item()
            
            train_dataset.df.loc[idxs, 'training_prob'] = probs

        metrics = get_metrics(probs, labels, threshold)
        avg_loss = epoch_loss / len(train_loader)
        
        return avg_loss, metrics


def tune(
    epoch,
    model,
    tune_dataset,
    criterion,
    batch_size,
    threshold=0.5,
    ):

    model.eval()
    epoch_loss = 0
    probs = []
    labels = []
    idxs = []

    tune_loader = torch.utils.data.DataLoader(
        tune_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    with tqdm.tqdm(
        tune_loader,
        desc=(f'Tune - Epoch {epoch}'),
        unit=' tensor',
        ncols=80,
        unit_scale=batch_size,
        position=1,
        leave=True) as t:

        with torch.no_grad():
            
            for i, batch in enumerate(t):

                idx, stacked_tiles, label = batch
                stacked_tiles, label = stacked_tiles.cuda(device='cuda:1'), label.cuda(device='cuda:0')
                logits = model(stacked_tiles)
                loss = criterion(logits, label.float())
                
                prob = torch.sigmoid(logits)
                probs.extend(prob[:,0].clone().tolist())
                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()
        
        tune_dataset.df.loc[idxs, 'validation_prob'] = probs

        metrics = get_metrics(probs, labels, threshold)
        avg_loss = epoch_loss / len(tune_loader)
        
        return avg_loss, metrics