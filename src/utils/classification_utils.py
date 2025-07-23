import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections.abc import Callable

from src.utils.metrics import get_metrics
from src.utils.train_utils import collate_features


def train(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    metric_names: list[str],
    batch_size: int = 1,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    gradient_accumulation: int | None = None,
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.train()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.RandomSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" case",
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):

            idx, x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, label)

            loss_value = loss.item()
            epoch_loss += loss_value

            if gradient_accumulation:
                loss = loss / gradient_accumulation

            loss.backward()

            if not gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()
            elif (i + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            pred = torch.topk(logits, 1, dim=1)[1]
            preds.extend(pred[:, 0].clone().tolist())

            prob = F.softmax(logits, dim=1).cpu().detach().numpy()
            probs = np.append(probs, prob, axis=0)

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, "pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
        probs,
    )

    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    metric_names: list[str],
    batch_size: int = 1,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.eval()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Tune"),
        unit=" case",
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for batch in t:

                idx, x, label = batch
                x = x.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits = model(x)
                loss = criterion(logits, label)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, "pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
        probs,
    )

    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def inference(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    metric_names: list[str],
    batch_size: int = 1,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.eval()
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Inference"),
        unit=" case",
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for batch in t:
                idx, x, label = batch
                x = x.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits = model(x)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, "pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
        probs,
    )

    results.update(metrics)

    return results