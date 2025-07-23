import tqdm
import torch
import torch.nn as nn

from functools import partial
from collections.abc import Callable

from src.utils.metrics import get_metrics
from src.utils.train_utils import collate_features_survival


def train(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    metric_names: list[str],
    batch_size: int = 1,
    collate_fn: Callable = partial(collate_features_survival, label_type="int"),
    gradient_accumulation: int | None = None,
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.train()
    epoch_loss = 0
    censoring, event_times, risk_scores = [], [], []
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

            idx, x, label, event_time, censored = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            censored = censored.to(device, non_blocking=True)

            logits = model(x) # [1, nbins]

            hazards = torch.sigmoid(logits)  # [1, nbins]
            surv = torch.cumprod(1 - hazards, dim=1)  # [1, nbins]

            loss = criterion(hazards, surv, label, censored)

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

            risk = -torch.sum(surv, dim=1).detach() # [1]
            risk_scores.append(risk.item())
            censoring.append(censored.item())
            event_times.append(event_time.item())

            idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "risk"] = risk_scores

    event_indicator = [bool(1 - c) for c in censoring]
    metrics = get_metrics(
        metric_names,
        risk_scores,
        event_times,
        event_indicator=event_indicator,
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
    collate_fn: Callable = partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.eval()
    epoch_loss = 0
    censoring, event_times, risk_scores = [], [], []
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

                idx, x, label, event_time, censored = batch
                x = x.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                censored = censored.to(device, non_blocking=True)

                logits = model(x)

                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1 - hazards, dim=1)

                loss = criterion(hazards, surv, label, censored, alpha=0)
                epoch_loss += loss.item()

                risk = -torch.sum(surv, dim=1).detach()
                risk_scores.append(risk.item())
                censoring.append(censored.item())
                event_times.append(event_time.item())

                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "risk"] = risk_scores

    event_indicator = [bool(1 - c) for c in censoring]
    metrics = get_metrics(
        metric_names,
        risk_scores,
        event_times,
        event_indicator=event_indicator,
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
    collate_fn: Callable = partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):

    model.eval()
    censoring, event_times, risk_scores = [], [], []
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
                idx, x, _, event_time, censored = batch
                x = x.to(device, non_blocking=True)
                censored = censored.to(device, non_blocking=True)

                logits = model(x)

                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1 - hazards, dim=1)

                risk = -torch.sum(surv, dim=1).detach()
                risk_scores.append(risk.item())
                censoring.append(censored.item())
                event_times.append(event_time.item())

                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "risk"] = risk_scores

    event_indicator = [bool(1 - c) for c in censoring]
    metrics = get_metrics(
        metric_names,
        risk_scores,
        event_times,
        event_indicator=event_indicator,
    )

    results.update(metrics)

    return results