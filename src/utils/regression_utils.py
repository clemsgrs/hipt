import tqdm
import torch
import torch.nn as nn

from functools import partial
from collections.abc import Callable

from src.utils.metrics import get_metrics
from src.utils.train_utils import collate_features


def get_pred_from_logits(logits: torch.Tensor, num_classes: int):
    pred = torch.max(
        torch.min(
            torch.round(logits),
            torch.Tensor([num_classes - 1])
        ),
        torch.Tensor([0]),
    )
    return pred


def train(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    metric_names: list[str],
    batch_size: int = 1,
    collate_fn: Callable = partial(collate_features, label_type="float"),
    gradient_accumulation: int | None = None,
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.train()
    epoch_loss = 0
    raw_logits, preds, labels = [], [], []
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
            loss = criterion(logits.squeeze(0), label) # erase batch dimension, which is always 1

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

            pred = get_pred_from_logits(logits.cpu(), dataset.num_classes)
            preds.extend(pred[:, 0].clone().tolist())

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))
            raw_logits.extend(logits.cpu().tolist())

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "raw_logit"] = raw_logits
    dataset.df.loc[idxs, "pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
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
    collate_fn: Callable = partial(collate_features, label_type="float"),
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.eval()
    epoch_loss = 0
    raw_logits, preds, labels = [], [], []
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

            for i, batch in enumerate(t):

                idx, x, label = batch
                x = x.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits = model(x)
                loss = criterion(logits.squeeze(0), label) # erase batch dimension, which is always 1

                pred = get_pred_from_logits(logits.cpu(), dataset.num_classes)
                preds.extend(pred[:, 0].clone().tolist())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))
                raw_logits.extend(logits.cpu().tolist())

                epoch_loss += loss.item()

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "raw_logit"] = raw_logits
    dataset.df.loc[idxs, "pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
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
    collate_fn: Callable = partial(collate_features, label_type="float"),
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.eval()
    raw_logits, preds, labels = [], [], []
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

                pred = get_pred_from_logits(logits.cpu(), dataset.num_classes)
                preds.extend(pred[:, 0].clone().tolist())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))
                raw_logits.extend(logits.cpu().tolist())

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "raw_logit"] = raw_logits
    dataset.df.loc[idxs, "pred"] = preds

    metrics = get_metrics(
        metric_names,
        preds,
        labels,
    )

    results.update(metrics)

    return results