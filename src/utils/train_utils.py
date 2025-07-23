import torch
import torch.nn as nn

from pathlib import Path

from src.utils.loss import NLLSurvLoss


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
    ):
        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif task == "regression":
            self.criterion = nn.MSELoss()
        elif task == "survival":
            self.criterion = NLLSurvLoss()

    def get_loss(self):
        return self.criterion


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
