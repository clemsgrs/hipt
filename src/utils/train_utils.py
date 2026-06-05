import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from src.utils.loss import NLLSurvLoss, CoxSurvLoss


def collate_features(batch, label_type: str = "int"):
    idx = [item[0] for item in batch]
    feature = torch.stack([item[1] for item in batch], dim=0)
    if label_type == "float":
        label = torch.FloatTensor([item[2] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[2] for item in batch])
    else:
        raise ValueError(f"Unsupported label_type: {label_type}")
    return [idx, feature, label]


def collate_features_survival(batch, label_type: str = "int"):
    idx = [item[0] for item in batch]
    feature = torch.stack([item[1] for item in batch], dim=0)
    if label_type == "float":
        label = torch.FloatTensor([item[2] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[2] for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    censored = torch.FloatTensor([item[4] for item in batch])
    return [idx, feature, label, event_time, censored]


class LossFactory:
    def __init__(
        self,
        task: str,
        survival_loss: str = "nll",
        cox_ties: str = "breslow",
    ):
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task == "regression":
            self.criterion = nn.MSELoss()
        elif task == "survival":
            if survival_loss == "coxph":
                self.criterion = CoxSurvLoss(ties=cox_ties)
            else:
                self.criterion = NLLSurvLoss()

    def get_loss(self):
        return self.criterion


class EventBalancedSampler(torch.utils.data.Sampler):
    """Flat index sampler whose every consecutive block of ``n`` indices contains
    at least ``min_events`` events.

    Bags are forwarded one at a time (batch_size=1), so this orders the *flat*
    stream of indices rather than returning batches. The Cox training loop slices
    the stream into windows of size ``n``; this sampler guarantees each window has
    events (a Cox loss needs >= 1 event to produce a gradient). It also drops the
    trailing partial window, so ``len`` is always a multiple of ``n``.

    This changes how patients are *grouped*, not which patients are seen: every
    event and censored patient is placed exactly once per epoch (up to the dropped
    remainder).
    """

    def __init__(
        self,
        censored,
        n: int,
        min_events: int = 1,
        seed: int = 0,
    ):
        censored = torch.as_tensor(np.asarray(censored).copy()).view(-1)
        self.event_idxs = torch.nonzero(censored == 0, as_tuple=False).view(-1).tolist()
        self.censored_idxs = torch.nonzero(censored == 1, as_tuple=False).view(-1).tolist()
        self.n = n
        self.min_events = max(1, min_events)
        self.seed = seed

        n_total = len(self.event_idxs) + len(self.censored_idxs)
        n_windows = n_total // n
        # cap by how many windows we can give >= min_events events
        n_windows = min(n_windows, len(self.event_idxs) // self.min_events)
        self.num_windows = max(0, n_windows)

    def set_epoch(self, epoch: int):
        self.seed = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        def shuffled(items):
            perm = torch.randperm(len(items), generator=g).tolist()
            return [items[i] for i in perm]

        events = shuffled(self.event_idxs)
        censored = shuffled(self.censored_idxs)

        windows = [[] for _ in range(self.num_windows)]
        ei = 0
        # deal min_events events into each window first
        for w in range(self.num_windows):
            for _ in range(self.min_events):
                windows[w].append(events[ei])
                ei += 1
        # fill the rest from the leftover events + all censored patients
        pool = shuffled(events[ei:] + censored)
        pi = 0
        for w in range(self.num_windows):
            while len(windows[w]) < self.n and pi < len(pool):
                windows[w].append(pool[pi])
                pi += 1

        flat = []
        for w in windows:
            flat.extend(w)
        return iter(flat)

    def __len__(self):
        return self.num_windows * self.n


class OptimizerFactory:
    def __init__(
        self,
        name: str,
        params: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        if name == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise KeyError(f"{name} not supported")

    def get_optimizer(self):
        return self.optimizer


class SchedulerFactory:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: dict | None = None,
    ):
        self.scheduler = None
        self.name = params.name
        if self.name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=params.step_size, gamma=params.gamma
            )
        elif self.name == "cosine":
            assert (
                params.T_max != -1
            ), "T_max parameter must be specified! If you dont know what to use, plug in nepochs"
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                params.T_max, eta_min=params.eta_min
            )
        elif self.name == "reduce_lr_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params.mode,
                factor=params.factor,
                patience=params.patience,
                min_lr=params.min_lr,
            )
        elif self.name:
            raise KeyError(f"{self.name} not supported")

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
        checkpoint_dir: Path | None = None,
        save_all: bool = False,
        verbose: bool = False,
    ):
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
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, epoch, model, results):
        score = results[self.tracking]
        if self.min_max == "min":
            score = -1 * score

        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            fname = f"best.pt"
            torch.save(model.state_dict(), Path(self.checkpoint_dir, fname))
            self.counter = 0

        elif score < self.best_score:
            self.counter += 1
            if epoch <= self.min_epoch + 1 and self.verbose:
                print(
                    f"EarlyStopping counter: {min(self.counter,self.patience)}/{self.patience}"
                )
            elif self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience and epoch > self.min_epoch:
                self.early_stop = True

        if self.save_all:
            fname = f"epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), Path(self.checkpoint_dir, fname))

        # override latest
        torch.save(model.state_dict(), Path(self.checkpoint_dir, "latest.pt"))
