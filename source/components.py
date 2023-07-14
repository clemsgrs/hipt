import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import Optional
from omegaconf import DictConfig

from source.utils import get_label_from_ordinal_label


class LossFactory:
    def __init__(
        self,
        task: str,
        loss: Optional[str] = None,
        label_encoding: Optional[str] = None,
        loss_options: Optional[DictConfig] = None,
    ):
        if task == "classification":
            if label_encoding == "ordinal":
                if loss == "ce":
                    self.criterion = nn.BCEWithLogitsLoss()
                elif loss == "coral":
                    self.criterion = CoralLoss()
                elif loss == "corn":
                    self.criterion = CornLoss(loss_options.num_classes)
            else:
                if loss == "ce":
                    self.criterion = nn.CrossEntropyLoss()
        elif task == "regression":
            self.criterion = nn.MSELoss()
        elif task == "survival":
            self.criterion = NLLSurvLoss()
            # self.criterion = NLLSurvLoss(alpha=loss_options.alpha)

    def get_loss(self):
        return self.criterion


def nll_loss(hazards, survival, Y, c, alpha=0.4, eps=1e-7):
    """
    Continuous time scale divided into k discrete bins: T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
    Y = T_discrete is the discrete event time:
        - Y = -1 if T_cont \in (-inf, 0)
        - Y = 0 if T_cont \in [0, a_1)
        - Y = 1 if T_cont in [a_1, a_2)
        - ...
        - Y = k-1 if T_cont in [a_(k-1), inf)
    hazards = discrete hazards, hazards(t) = P(Y=t | Y>=t, X) for t = -1, 0, 1, 2, ..., k-1
    survival = survival function, survival(t) = P(Y > t | X)

    All patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
    -> hazards(-1) = 0
    -> survival(-1) = P(Y > -1 | X) = 1

    Summary:
        - neural network is hazard probability function, h(t) for t = 0, 1, 2, ..., k-1
        - h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 0, 1, 2, ..., k-1
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if survival is None:
        survival = torch.cumprod(
            1 - hazards, dim=1
        )  # survival is cumulative product of 1 - hazards
    survival_padded = torch.cat(
        [torch.ones_like(c), survival], 1
    )  # survival(-1) = 1, all patients are alive from (-inf, 0) by definition
    # after padding, survival(t=-1) = survival[0], survival(t=0) = survival[1], survival(t=1) = survival[2], etc
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(survival_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(
        torch.gather(survival_padded, 1, Y + 1).clamp(min=eps)
    )
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, label, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, label, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, label, c, alpha=alpha)


def coral_loss(logits, ordinal_labels, importance_weights=None, reduction="mean"):
    """
    Computes the CORAL loss described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        model output.

    ordinal_labels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors.

    importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
        Optional weights for the different labels in levels.
        A tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as None.

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).
    """

    if not logits.shape == ordinal_labels.shape:
        raise ValueError(
            f"Please ensure that logits ({logits.shape}) has the same shape as levels ({ordinal_labels.shape})."
        )

    # original implementation
    # loss_term = F.logsigmoid(logits)*ordinal_labels + (F.logsigmoid(logits) - logits)*(1-ordinal_labels)

    # my implementation
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loss_term = -criterion(logits, ordinal_labels)

    if importance_weights is not None:
        loss_term *= importance_weights

    val = -torch.sum(loss_term, dim=1)

    if reduction == "mean":
        loss = torch.mean(val)
    elif reduction == "sum":
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        raise ValueError(
            f'Invalid value for `reduction`. Should be "mean", "sum", or None. Got {reduction}'
        )

    return loss


class CoralLoss(torch.nn.Module):
    """
    Computes the CORAL loss described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, ordinal_labels, importance_weights=None):
        return coral_loss(
            logits,
            ordinal_labels,
            importance_weights=importance_weights,
            reduction=self.reduction,
        )


def corn_loss(logits, ordinal_labels, num_classes):
    """
    Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        model output.

    ordinal_labels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors.

    num_classes : int
        Number of unique class labels (class labels should start at 0).

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.
    """

    # transform ordinal labels back to regular class labels
    class_labels = get_label_from_ordinal_label(ordinal_labels)

    sets = []
    for i in range(num_classes - 1):
        # identify the samples whose label is at least i
        mask = class_labels > i - 1
        # among these samples, identify the ones whose label is at least i+1
        label_tensor = (class_labels[mask] > i).to(torch.int64)
        sets.append((mask, label_tensor))

    num_examples = 0
    total_loss = 0.0
    for class_idx, s in enumerate(sets):
        mask = s[0]
        labels = s[1]

        if len(labels) < 1:
            continue

        num_examples += len(labels)
        pred = logits[mask, class_idx]

        # original implementation
        # loss_term = F.logsigmoid(pred)*labels + (F.logsigmoid(pred) - pred)*(1-labels)
        # loss = -torch.sum(loss_term, dim=1)

        # my implementation
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
        loss = criterion(pred, labels.float())

        total_loss += loss

    return total_loss / num_examples


class CornLoss(torch.nn.Module):
    """
    Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    num_classes : int
        Number of unique class labels (class labels should start at 0).
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, ordinal_labels):
        return corn_loss(logits, ordinal_labels, num_classes=self.num_classes)


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.distributed = torch.cuda.device_count() > 1

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        m = 1
        if self.distributed:
            dist.all_reduce(batch_center)
            m = dist.get_world_size()
        batch_center = batch_center / (len(teacher_output) * m)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
