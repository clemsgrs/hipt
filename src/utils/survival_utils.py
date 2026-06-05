import tqdm
import torch
import torch.nn as nn

from functools import partial
from collections.abc import Callable

from src.utils.metrics import get_metrics
from src.utils.train_utils import collate_features_survival, EventBalancedSampler


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


def train_coxph(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    metric_names: list[str],
    n: int = 48,
    event_balanced: bool = True,
    min_events: int = 1,
    collate_fn: Callable = partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """Cox training with prediction accumulation.

    Each window forwards ``n`` single-patient bags one at a time, keeps their
    graph-connected risk scalars, computes one Cox loss over the window, and
    backpropagates once. See cox-ph-trick.md for the rationale.
    """

    model.train()
    epoch_loss = 0.0
    num_windows = 0
    censoring, event_times, risk_scores = [], [], []
    idxs = []

    n_events = int((dataset.df.censored.values == 0).sum())
    if event_balanced:
        sampler = EventBalancedSampler(
            dataset.df.censored.values, n=n, min_events=min_events, seed=epoch
        )
        if sampler.num_windows == 0:
            raise ValueError(
                f"cox.n={n} (min_events={min_events}) yields 0 windows for "
                f"{len(dataset)} patients / {n_events} events. "
                f"Lower cox.n or cox.min_events."
            )
    else:
        if len(dataset) < n:
            raise ValueError(
                f"cox.n={n} exceeds dataset size ({len(dataset)}); no window can "
                f"be filled. Lower cox.n."
            )
        sampler = torch.utils.data.RandomSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    results = {}

    risk_buffer, time_buffer, cens_buffer = [], [], []

    def flush():
        nonlocal epoch_loss, num_windows
        if len(risk_buffer) == 0:
            return
        risks = torch.cat(risk_buffer, dim=0).view(-1)
        times = torch.cat(time_buffer, dim=0).view(-1)
        cens = torch.cat(cens_buffer, dim=0).view(-1)

        loss = criterion(risks, times, cens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_windows += 1
        risk_buffer.clear()
        time_buffer.clear()
        cens_buffer.clear()

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" case",
        leave=False,
    ) as t:
        for batch in t:
            idx, x, _, event_time, censored = batch
            x = x.to(device, non_blocking=True)
            event_time = event_time.to(device, non_blocking=True)
            censored = censored.to(device, non_blocking=True)

            risk = model(x).view(1)  # [1], raw scalar = log relative hazard

            risk_buffer.append(risk)
            time_buffer.append(event_time.view(1))
            cens_buffer.append(censored.view(1))

            risk_scores.append(risk.item())
            censoring.append(censored.item())
            event_times.append(event_time.item())
            idxs.extend(list(idx))

            if len(risk_buffer) == n:
                flush()

    # the event-balanced sampler yields a multiple of n, so nothing is left over;
    # under random sampling we drop the trailing partial window (no event guarantee)
    risk_buffer.clear()
    time_buffer.clear()
    cens_buffer.clear()

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

    results["loss"] = epoch_loss / max(num_windows, 1)
    return results


def _coxph_eval(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    metric_names: list[str],
    criterion: Callable | None = None,
    desc: str = "Tune",
    epoch: int | None = None,
    collate_fn: Callable = partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):
    """Shared Cox evaluation: collect raw risks over the full cohort under no_grad,
    then compute one Cox loss over the entire cohort (the true full risk set)."""

    model.eval()
    censoring, event_times, risk_scores = [], [], []
    idxs = []

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=torch.utils.data.SequentialSampler(dataset),
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    label = f"Epoch {epoch} - {desc}" if epoch is not None else desc
    with tqdm.tqdm(loader, desc=label, unit=" case", leave=(epoch is None)) as t:
        with torch.no_grad():
            for batch in t:
                idx, x, _, event_time, censored = batch
                x = x.to(device, non_blocking=True)

                risk = model(x).view(1)
                risk_scores.append(risk.item())
                censoring.append(censored.item())
                event_times.append(event_time.item())
                idxs.extend(list(idx))

    assert len(idxs) == len(set(idxs)), "idxs must be unique"
    dataset.df.loc[idxs, "risk"] = risk_scores

    event_indicator = [bool(1 - c) for c in censoring]
    results = get_metrics(
        metric_names,
        risk_scores,
        event_times,
        event_indicator=event_indicator,
    )

    if criterion is not None:
        risks = torch.tensor(risk_scores, dtype=torch.float32, device=device)
        times = torch.tensor(event_times, dtype=torch.float32, device=device)
        cens = torch.tensor(censoring, dtype=torch.float32, device=device)
        results["loss"] = criterion(risks, times, cens).item()

    return results


def tune_coxph(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    metric_names: list[str],
    batch_size: int = 1,  # ignored: eval is always sequential, one bag at a time
    collate_fn: Callable = partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):
    return _coxph_eval(
        model,
        dataset,
        metric_names,
        criterion=criterion,
        desc="Tune",
        epoch=epoch,
        collate_fn=collate_fn,
        num_workers=num_workers,
        device=device,
    )


def inference_coxph(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    metric_names: list[str],
    batch_size: int = 1,  # ignored: eval is always sequential, one bag at a time
    collate_fn: Callable = partial(collate_features_survival, label_type="int"),
    num_workers: int = 0,
    device: torch.device | None = None,
):
    return _coxph_eval(
        model,
        dataset,
        metric_names,
        criterion=None,
        desc="Inference",
        epoch=None,
        collate_fn=collate_fn,
        num_workers=num_workers,
        device=device,
    )