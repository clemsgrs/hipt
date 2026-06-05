# CoxPH Training with Prediction Accumulation for Variable-Size MIL Bags

> Revised after design review. This supersedes the original proposal. Where this
> document diverges from the naive approach, the reasoning is called out inline.

## Goal

Add an **opt-in** continuous-time CoxPH survival training mode for MIL aggregators,
alongside the existing discrete-time NLL path (`NLLSurvLoss`). Each patient/slide
bag is processed with `batch_size=1` (bags have variable tile counts), but the Cox
partial-likelihood loss is computed over an effective mini-batch of multiple patients.

The existing NLL path (`task == "survival"`, model outputs `[1, nbins]`, risk =
`-sum(surv)`) is **left untouched**. Cox is purely additive.

## Why prediction accumulation (and why NLL does *not* need it)

The current `NLLSurvLoss` is **per-patient independent**: each patient contributes a
self-contained loss term, so `batch_size=1` plus ordinary gradient accumulation
already works. CoxPH is different — the partial likelihood couples patients through a
**risk set** (each event's denominator sums over everyone still at risk). A single
patient has no risk set, so standard gradient accumulation (one independent loss per
microbatch) is meaningless for Cox.

The fix: forward several single-patient bags, keep their **graph-connected** risk
scalars, concatenate them, compute **one** Cox loss over the group, and backprop once.

```python
# correct pattern (one "window" = N forwards -> 1 cat -> 1 loss -> 1 backward)
risks = []
for _ in range(N):
    risks.append(model(bag).view(1))   # NO detach / item / no_grad
loss = cox_loss(torch.cat(risks), times, events)
loss.backward()
optimizer.step()
```

Do **not** call `.detach()`, `.item()`, `.cpu().numpy()`, or wrap the forward in
`torch.no_grad()` during training — risk tensors must stay connected to their graph
until `loss.backward()`.

## Key design decisions (review outcomes)

### 1. Use a library for the Cox loss — do not hand-roll it

Use **`torchsurv`** (`torchsurv.loss.cox.neg_partial_log_likelihood`). It is pure
PyTorch, differentiable, and handles tie approximations (Breslow / Efron) via a flag.
This removes the need to implement sorting, `logcumsumexp`, and tie handling ourselves.

- New dependency: pin `torchsurv==0.1.6` in requirements.
- Keep `scikit-survival` (`sksurv`) for the C-index metric — it is numpy/CPU and not
  differentiable, so it is for evaluation only, not the loss. Already in use in
  `src/utils/metrics.py`; no change needed there beyond the sign note below.

**Polarity trap:** the dataset yields `censored` (1 = censored). `torchsurv` expects
`event` (1 = event). Pass `event = 1 - censored`. Verify the exact argument order and
boolean/float expectations against the installed 0.1.6 API at implementation time — a
flipped convention silently trains the model to rank survival backwards.

**fp32:** compute the Cox loss in float32 even under AMP (numerical stability of the
log-cumsum term).

### 2. Risk-set is a random sub-cohort — accept the approximation, raise N

Computing the Cox loss over an N-patient window optimizes an expectation over random
N-patient sub-cohorts, not the full-cohort partial likelihood. The gradient is biased
relative to full-cohort Cox; the bias shrinks as N grows. This is the standard
"Cox loss on mini-batches" technique and is accepted here.

Because tile features are **precomputed/frozen** and the aggregator is light, holding N
forward-graphs is cheap, so prefer a **moderately large N (32–64)** rather than the
small N=16 in the original draft. Larger N = larger risk set = less biased gradient.

### 3. Event-balanced sampling — not "skip no-event windows"

With random sampling and heavy censoring (common in pathology), some windows contain
zero events and produce no gradient. The naive fix (compute the loss, then skip the
window if `events.sum() == 0`) **wastes N forward passes** and **biases sampling** by
systematically discarding the most-censored draws. (It also had dead code: a
graph-connected zero-loss return that was never reached because the caller bailed
before `backward()`.)

Instead, use a **custom `BatchSampler`** that constructs each window to contain **≥1
event** (ideally several). This eliminates wasted windows and lowers gradient variance.
The sampler needs per-`idx` `censored` access from the dataset.

### 4. Model head: single risk scalar

When Cox is selected, build the model with `num_classes = 1` and use the raw scalar
logit directly as the risk (higher = higher hazard / shorter survival). Assert
`num_classes == 1` when the Cox loss is selected. Do **not** reuse the binned head or
the `-sum(surv)` derivation — those belong to the NLL path only.

### 5. Evaluation: full-cohort Cox loss, no accumulation

Prediction accumulation exists only to bound **training** memory. At eval there is no
backward pass and no graph to hold, so run all tune/test patients under
`torch.no_grad()`, collect every risk scalar, and compute the Cox loss over the
**entire cohort at once** — the true full risk set, which is *more* correct than
training and **deterministic across epochs** (a clean early-stopping signal).

Update the Cox branch of `tune` / `inference` accordingly:

- **risk** = raw model output (no sigmoid / cumprod / surv).
- **validation loss** = one `neg_partial_log_likelihood` over all tune patients.
- **C-index** = `concordance_index_censored(event_indicator, event_time, risk)` —
  unchanged call, but feed the **raw Cox risk** (higher = worse). Do not apply the
  NLL `-sum(surv)` sign flip; that would double-invert.

Early stopping continues to watch **tune loss** (`min`); under Cox this is the
full-cohort partial likelihood.

### 6. One accumulation axis only (Lever 1)

Two orthogonal levers exist:

- **Lever 1 — N (prediction accumulation):** number of forward-graphs held before one
  backward = the risk-set size of a single Cox loss. Bounded by memory. **This is the
  only lever that improves the risk-set approximation.**
- **Lever 2 — gradient accumulation across windows (M):** run M separate windows, each
  with its own Cox loss and `backward()`, summing `.grad` before one `optimizer.step()`.
  This lowers gradient variance and reduces step frequency but does **not** enlarge the
  risk set — M windows give M independent denominators of size N, never one of size M·N.

Decision: **implement Lever 1 only**, one `optimizer.step()` per window. Since memory
is cheap, just raise N directly. Lever 2 is deferred — it only earns its keep if we hit
a memory wall, can't raise N further, and still want lower-variance steps.

### 7. Learning rate / schedule consequence

With one optimizer step per N patients, there are ~N× fewer updates per epoch than the
per-patient NLL path. Budget for a **higher LR and/or more epochs** in the sweep, or
training will appear to stall.

## Codebase touch points

- `src/utils/train_utils.py::LossFactory` — add a Cox branch (currently maps
  `task == "survival"` → `NLLSurvLoss`). Decide Cox selection via a config field
  (e.g. `survival.loss: coxph`) rather than overloading `task`.
- `src/utils/survival_utils.py::train` — add the prediction-accumulation loop for the
  Cox path (the current loop does per-patient NLL with optional gradient accumulation).
- `src/utils/survival_utils.py::tune` / `inference` — add the full-cohort Cox eval path
  (replace `-sum(surv)` risk with raw output; compute full-cohort Cox loss).
- Model construction (`ModelFactory`) — `num_classes = 1` for Cox.
- Dataset already yields `(idx, x, label, event_time, censored)`; Cox consumes
  `event_time` and `1 - censored` directly and **ignores the discrete `label` bin**.
- New custom `BatchSampler` for event-balanced windows.
- `requirements` — add `torchsurv==0.1.6`.
- `src/utils/metrics.py` — no change to `concordance_index_censored`; just ensure the
  Cox caller passes raw risk.

## Suggested config

```yaml
survival:
  loss: coxph              # selects the Cox path; default stays nll
  cox:
    n: 48                  # prediction-accumulation window size = risk-set size
    ties: breslow          # or efron (torchsurv flag)
    event_balanced: true   # guarantee >= 1 event per window
    min_events_per_window: 1
batch_size: 1
```

Adapt names to the existing config structure.

## Open items (sweep, not blockers)

- **N**: 32–64 candidates; depends on cohort size and memory headroom.
- **Tie method**: Breslow if event times are granular (days) with few ties; Efron if
  coarse (months/years) with heavy ties.
- **Censoring rate**: drives how aggressive the event-balanced sampler must be.

## Loss math reference (informational — implemented by torchsurv)

For the no-ties case the Breslow partial likelihood is: sort by descending time, take
`logcumsumexp` over sorted risks (the cumulative term at position *i* equals the
log-sum-exp over all patients with time ≥ time_i, i.e. the risk set), and for event
patients minimize `-(risk_i - logcumsumexp_i)`, averaged over events. We rely on
`torchsurv` for this plus Efron tie handling; we do not reimplement it.

## Tests to add

1. **Graph connectivity** — after N forwards + one Cox `backward()`, model params have
   non-zero gradients.
2. **No-detach** — accumulated risks satisfy `requires_grad` and `grad_fn is not None`
   before the loss.
3. **Polarity** — a toy cohort where higher risk on early-event patients yields lower
   loss than the reversed assignment (guards the `event = 1 - censored` mapping and the
   C-index sign together).
4. **Event-balanced sampler** — every emitted window contains ≥ `min_events_per_window`
   events; no window is ever empty of events.
5. **Full-cohort eval** — eval risk equals the raw model output and the eval Cox loss is
   computed once over all patients (not windowed).

## Acceptance criteria

1. Can train a variable-size MIL survival model with `batch_size=1`, Cox loss, and
   `n > 1`, with the NLL path unchanged.
2. Cox loss is computed once per window (one optimizer step per window).
3. Risk tensors stay graph-connected until the loss is computed.
4. Every training window contains at least one event (event-balanced sampler).
5. Eval computes a single full-cohort Cox loss and uses raw risk for the C-index.
6. Tests cover graph connectivity, no-detach, polarity/sign, sampler guarantees, and
   full-cohort eval.
7. Compatible with the existing training-loop style and config system.
```