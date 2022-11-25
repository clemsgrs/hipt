import tqdm
import wandb
import torch
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from typing import Optional, Callable, List, Dict
from sklearn import metrics
from sklearn.model_selection import train_test_split


def initialize_wandb(project, entity, exp_name, dir='./wandb', config={}, tags=None, key=''):
    dir = Path(dir)
    command = f'wandb login {key}'
    subprocess.call(command, shell=True)
    if tags == None:
        tags=[]
    run = wandb.init(project=project, entity=entity, name=exp_name, dir=dir, config=config, tags=tags)
    return run


def initialize_df(slide_ids):
    nslide = len(slide_ids)
    df_dict = {
        'slide_id': slide_ids,
        'process': np.full((nslide), 1, dtype=np.uint8),
        'status': np.full((nslide), 'tbp'),
    }
    df = pd.DataFrame(df_dict)
    return df


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
    train_df, tune_df = train_test_split(df, test_size=tune_size, random_state=seed, stratify=df.label)
    test_df = pd.DataFrame()
    if test_size > 0:
        train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=seed, stratify=train_df.label)
        test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    tune_df = tune_df.reset_index(drop=True)
    if save_csv:
        train_df.to_csv(Path(output_dir, f'train.csv'), index=False)
        tune_df.to_csv(Path(output_dir, f'tune.csv'), index=False)
        if test_size > 0:
            test_df.to_csv(Path(output_dir, f'test.csv'), index=False)
    return train_df, tune_df, test_df


def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_binary_metrics(probs: np.array(float), preds: List[int], labels: List[int]):
    labels = np.asarray(labels)
    acc = metrics.accuracy_score(labels, preds)
    auc = metrics.roc_auc_score(labels, probs)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds)
    metrics_dict = {'accuracy': acc, 'auc': auc, 'precision': precision, 'recall': recall}
    return metrics_dict


def get_metrics(probs: np.array(float), preds: List[int], labels: List[int], multi_class: str = 'ovr'):
    labels = np.asarray(labels)
    auc = metrics.roc_auc_score(labels, probs, multi_class=multi_class)
    quadratic_weighted_kappa = metrics.cohen_kappa_score(labels, preds, weights='quadratic')
    metrics_dict = {'auc': auc, 'kappa': quadratic_weighted_kappa}
    return metrics_dict


def collate_features(batch, label_type: str = 'int'):
    idx = torch.LongTensor([item[0] for item in batch])
    # feature = torch.vstack([item[1] for item in batch])
    feature = torch.cat([item[1] for item in batch], dim=0)
    if label_type == 'float':
        label = torch.FloatTensor([item[2] for item in batch])
    elif label_type == 'int':
        label = torch.LongTensor([item[2] for item in batch])
    return [idx, feature, label]


def collate_region_filepaths(batch):
    item = batch[0]
    idx = torch.LongTensor([item[0]])
    fp = item[1]
    return [idx, fp]


def get_roc_auc_curve(probs: np.array(float), labels: List[int]):
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    auc = metrics.roc_auc_score(labels, probs)
    fig = plt.figure(dpi=600)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.legend(loc='lower right')
    img = wandb.Image(fig)
    plt.close()
    return img


def log_on_step(name, results, step: str = 'epoch', to_log: Optional[List['str']] = None):
    if not to_log:
        to_log = list(results.keys())
    for r, v in results.items():
        if r in to_log:
            wandb.define_metric(f'{name}/{r}', step_metric=step)
            wandb.log({f'{name}/{r}': v})


def make_weights_for_balanced_classes(dataset):
    n_samples = len(dataset)
    weight_per_class = []
    for c in range(dataset.num_classes):
        w = n_samples * 1. / len(dataset.class_2_id[c])
        weight_per_class.append(w)
    weight = []
    for idx in range(len(dataset)):
        y = dataset.get_label(idx)
        weight.append(weight_per_class[y])
    return torch.DoubleTensor(weight)


def logit_to_ordinal_prediction(logits):
    with torch.no_grad():
        pred = torch.sigmoid(logits)
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


class OptimizerFactory:

    def __init__(
        self,
        name: str,
        params: nn.Module,
        lr: float,
        weight_decay: float = 0.,
        momentum: float = 0.,
        ):

            if name == 'adam':
                self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            elif name == 'sgd':
                self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
            else:
                raise KeyError(f'{name} not supported')

    def get_optimizer(self):
        return self.optimizer


class SchedulerFactory:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: Optional[dict] = None,
    ):

        self.scheduler = None
        self.name = params.name
        if self.name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
        elif self.name == 'cosine':
            assert params.T_max != -1, 'T_max parameter must be specified! If you dont know what to use, plug in nepochs'
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(params.T_max, eta_min=params.eta_min)
        elif self.name == 'reduce_lr_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=params.mode, factor=params.factor, patience=params.patience, min_lr=params.min_lr)
        elif self.name:
            raise KeyError(f'{self.name} not supported')

    def get_scheduler(self):
        return self.scheduler


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(
        self,
        tracking: str,
        min_max: str,
        patience: int = 20,
        min_epoch: int = 50,
        checkpoint_dir: Optional[Path] = None,
        save_all: bool = False,
        verbose: bool = False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement
        """
        self.tracking = tracking
        self.min_max = min_max
        self.patience = patience
        self.min_epoch = min_epoch
        self.checkpoint_dir = checkpoint_dir
        self.save_all = save_all
        self.verbose = verbose

        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, model, results):

        score = results[self.tracking]
        if self.min_max == 'min':
            score = -1 * score

        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            fname = f'best_model_{wandb.run.id}.pt'
            torch.save(model.state_dict(), Path(self.checkpoint_dir, fname))
            self.counter = 0

        elif score < self.best_score:
            self.counter += 1
            if epoch <= self.min_epoch+1 and self.verbose:
                print(f'EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}')
            elif self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience and epoch > self.min_epoch:
                self.early_stop = True

        if self.save_all:
            fname = f'epoch_{epoch}.pt'
            torch.save(model.state_dict(), Path(self.checkpoint_dir, fname))


def train(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features, label_type='int'),
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
    gradient_clipping: Optional[int] = None,
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    epoch_loss = 0
    probs = np.empty((0,dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.RandomSampler(dataset)
    if weighted_sampling:
        weights = make_weights_for_balanced_classes(dataset)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            len(weights),
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f'Train - Epoch {epoch}'),
        unit=' slide',
        ncols=80,
        unit_scale=batch_size,
        leave=True) as t:

        for i, batch in enumerate(t):

            optimizer.zero_grad()
            idx, x, label = batch
            x, label = x.to(device, non_blocking=True), label.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, label)

            loss_value = loss.item()
            epoch_loss += loss_value

            if gradient_clipping:
                loss = loss / gradient_clipping

            loss.backward()
            optimizer.step()

            pred = torch.topk(logits, 1, dim=1)[1]
            preds.extend(pred[:,0].clone().tolist())

            prob = F.softmax(logits, dim=1).cpu().detach().numpy()
            probs = np.append(probs, prob, axis=0)

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    #TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f'prob_{class_idx}'] = p.tolist()

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(probs[:,1], preds, labels)
        roc_auc_curve = get_roc_auc_curve(probs[:,1], labels)
        results.update({'roc_auc_curve': roc_auc_curve})
    else:
        metrics = get_metrics(probs, preds, labels)

    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results['loss'] = train_loss

    return results


def tune(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features, label_type='int'),
    batch_size: Optional[int] = 1,
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    epoch_loss = 0
    probs = np.empty((0,dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f'Tune - Epoch {epoch}'),
        unit=' slide',
        ncols=80,
        unit_scale=batch_size,
        leave=True) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, label)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:,0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()

    #TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f'prob_{class_idx}'] = p.tolist()

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(probs[:,1], preds, labels)
        roc_auc_curve = get_roc_auc_curve(probs[:,1], labels)
        results.update({'roc_auc_curve': roc_auc_curve})
    else:
        metrics = get_metrics(probs, preds, labels)

    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results['loss'] = tune_loss

    return results


def test(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Callable = partial(collate_features, label_type='int'),
    batch_size: Optional[int] = 1,
    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    probs = np.empty((0,dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f'Test'),
        unit=' slide',
        ncols=80,
        unit_scale=batch_size,
        leave=True) as t:

        with torch.no_grad():

            for i, batch in enumerate(t):

                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(device, non_blocking=True)
                logits = model(x)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:,0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    #TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f'prob_{class_idx}'] = p.tolist()

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(probs[:,1], preds, labels)
        roc_auc_curve = get_roc_auc_curve(probs[:,1], labels)
        results.update({'roc_auc_curve': roc_auc_curve})
    else:
        metrics = get_metrics(probs, preds, labels)

    results.update(metrics)

    return results