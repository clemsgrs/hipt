"""Unit tests for the CoxPH prediction-accumulation training path.

Run with: pytest tests/test_coxph.py
"""

import torch
import torch.nn as nn

from src.utils.loss import CoxSurvLoss
from src.utils.train_utils import EventBalancedSampler


def _toy_model():
    # maps a variable-size bag [1, k, d] to a single risk scalar [1, 1]
    return nn.Sequential(
        nn.Linear(4, 1),
    )


def _bag(k):
    return torch.randn(k, 4)


def test_graph_connectivity_and_no_detach():
    """Params receive non-zero gradients after N forwards + one Cox loss backward,
    and accumulated risks stay graph-connected until the loss."""
    torch.manual_seed(0)
    head = _toy_model()

    risk_buffer = []
    for k in (3, 7, 5, 2):  # variable-size bags
        pooled = head(_bag(k)).mean(dim=0)  # [1] raw risk scalar
        risk_buffer.append(pooled.view(1))

    risks = torch.cat(risk_buffer, dim=0)
    # no-detach: accumulated risks must still be connected to the graph
    assert risks.requires_grad
    assert risks.grad_fn is not None

    times = torch.tensor([1.0, 2.0, 3.0, 4.0])
    censored = torch.tensor([0.0, 0.0, 1.0, 1.0])  # first two are events

    loss = CoxSurvLoss(ties="breslow")(risks, times, censored)
    loss.backward()

    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_cox_loss_polarity():
    """Event patients with higher predicted risk should give lower loss than the
    reversed assignment. Also guards the censored -> event mapping."""
    times = torch.tensor([1.0, 2.0, 3.0, 4.0])
    censored = torch.tensor([0.0, 0.0, 1.0, 1.0])  # early patients are events

    good = torch.tensor([4.0, 3.0, 2.0, 1.0])  # high risk for early-event
    bad = torch.tensor([1.0, 2.0, 3.0, 4.0])  # reversed

    loss = CoxSurvLoss(ties="breslow")
    assert loss(good, times, censored).item() < loss(bad, times, censored).item()


def test_event_balanced_sampler_guarantees_events():
    """Every emitted window contains >= min_events events, the length is a multiple
    of n, and no index is duplicated within an epoch."""
    n = 8
    min_events = 2
    # 40 patients, 12 events (event = censored 0)
    censored = torch.ones(40)
    censored[:12] = 0.0
    event_set = set(range(12))

    sampler = EventBalancedSampler(censored, n=n, min_events=min_events, seed=1)
    flat = list(iter(sampler))

    assert len(flat) == len(sampler)
    assert len(flat) % n == 0
    assert len(flat) == len(set(flat)), "indices must not repeat within an epoch"

    for w in range(0, len(flat), n):
        window = flat[w : w + n]
        n_events = sum(1 for i in window if i in event_set)
        assert n_events >= min_events


def test_event_balanced_sampler_reseeds_per_epoch():
    censored = torch.ones(40)
    censored[:12] = 0.0
    sampler = EventBalancedSampler(censored, n=8, min_events=1, seed=0)

    sampler.set_epoch(0)
    order_a = list(iter(sampler))
    sampler.set_epoch(1)
    order_b = list(iter(sampler))

    assert order_a != order_b, "different epochs should produce different windows"


def test_full_cohort_eval_loss_is_scalar():
    """The eval Cox loss is computed once over the whole cohort, not per window."""
    torch.manual_seed(0)
    risks = torch.randn(50)
    times = torch.rand(50) * 10
    censored = (torch.rand(50) > 0.5).float()
    loss = CoxSurvLoss(ties="breslow")(risks, times, censored)
    assert loss.dim() == 0  # single scalar over the full cohort
