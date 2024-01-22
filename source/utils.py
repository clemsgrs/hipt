import tqdm
import math
import wandb
import torch
import random
import matplotlib
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from collections import Counter
from collections.abc import Callable
from typing import Optional, List, Union
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_device(gpu_id: int):
    if gpu_id == -1 and torch.cuda.is_available():
        device_name = "cuda"
    elif torch.cuda.is_available():
        device_name = f"cuda:{gpu_id}"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    return device


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def write_dictconfig(d, f, child: bool = False, ntab=0):
    for k, v in d.items():
        if isinstance(v, dict):
            if not child:
                f.write(f"{k}:\n")
            else:
                for _ in range(ntab):
                    f.write("\t")
                f.write(f"- {k}:\n")
            write_dictconfig(v, f, True, ntab=ntab + 1)
        else:
            if isinstance(v, list):
                if not child:
                    f.write(f"{k}:\n")
                    for e in v:
                        f.write(f"\t- {e}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"{k}:\n")
                    for e in v:
                        for _ in range(ntab):
                            f.write("\t")
                        f.write(f"\t- {e}\n")
            else:
                if not child:
                    f.write(f"{k}: {v}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"- {k}: {v}\n")


def initialize_wandb(
    cfg: DictConfig,
    key: Optional[str] = "",
):
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if cfg.wandb.tags == None:
        tags = []
    else:
        tags = [str(t) for t in cfg.wandb.tags]
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.wandb.resume_id:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
            id=cfg.wandb.resume_id,
            resume="must",
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
        )
    config_file_path = Path(run.dir, "run_config.yaml")
    d = OmegaConf.to_container(cfg, resolve=True)
    with open(config_file_path, "w+") as f:
        write_dictconfig(d, f)
        wandb.save(str(config_file_path))
        f.close()
    return run


def initialize_df(slide_ids):
    nslide = len(slide_ids)
    df_dict = {
        "slide_id": slide_ids,
        "process": np.full((nslide), 1, dtype=np.uint8),
        "status": np.full((nslide), "tbp"),
    }
    df = pd.DataFrame(df_dict)
    return df


def extract_coord_from_path(path):
    """
    Path expected to look like /path/to/dir/x_y.png
    """
    x_y = path.stem
    x, y = x_y.split("_")[0], x_y.split("_")[1]
    return int(x), int(y)


def update_state_dict(model_dict, state_dict):
    success, failure = 0, 0
    updated_state_dict = {}
    for k, v in zip(model_dict.keys(), state_dict.values()):
        if v.size() != model_dict[k].size():
            updated_state_dict[k] = model_dict[k]
            failure += 1
        else:
            updated_state_dict[k] = v
            success += 1
    msg = f"{success} weight(s) loaded succesfully ; {failure} weight(s) not loaded because of mismatching shapes"
    return updated_state_dict, msg


def create_train_tune_test_df(
    df: pd.DataFrame,
    save_csv: bool = False,
    output_dir: Path = Path(""),
    tune_size: float = 0.4,
    test_size: float = 0.2,
    seed: Optional[int] = 21,
):
    train_df, tune_df = train_test_split(
        df, test_size=tune_size, random_state=seed, stratify=df.label
    )
    test_df = pd.DataFrame()
    if test_size > 0:
        train_df, test_df = train_test_split(
            train_df, test_size=test_size, random_state=seed, stratify=train_df.label
        )
        test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    tune_df = tune_df.reset_index(drop=True)
    if save_csv:
        train_df.to_csv(Path(output_dir, f"train.csv"), index=False)
        tune_df.to_csv(Path(output_dir, f"tune.csv"), index=False)
        if test_size > 0:
            test_df.to_csv(Path(output_dir, f"test.csv"), index=False)
    return train_df, tune_df, test_df


def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_binary_metrics(
    preds: List[int], labels: List[int], probs: Optional[np.array] = None
):
    labels = np.asarray(labels)
    acc = metrics.accuracy_score(labels, preds)
    if probs is not None:
        auc = metrics.roc_auc_score(labels, probs)
    else:
        auc = -1.0
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds)
    metrics_dict = {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
    }
    return metrics_dict


def get_metrics(
    preds: List[int],
    labels: List[int],
    probs: Optional[np.array] = None,
    multi_class: str = "ovr",
    class_names: Optional[List[str]] = None,
    use_wandb: bool = False,
):
    labels = np.asarray(labels)
    quadratic_weighted_kappa = metrics.cohen_kappa_score(
        labels, preds, weights="quadratic"
    )
    cm = plot_confusion_matrix(
        labels,
        preds,
        show_pct=True,
        cbar=False,
        names=class_names,
        normalize="true",
        title="Confusion Matrix",
    )
    metrics_dict = {"kappa": quadratic_weighted_kappa, "cm": cm}
    if use_wandb:
        wandb_cm = wandb.plot.confusion_matrix(
            y_true=labels,
            preds=preds,
            class_names=class_names,
        )
        metrics_dict.update({"wandb_cm": wandb_cm})
    if probs is not None:
        auc = metrics.roc_auc_score(labels, probs, multi_class=multi_class)
        metrics_dict.update({"auc": auc})
    return metrics_dict


def collate_features(batch, label_type: str = "int"):
    idx = torch.LongTensor([item[0] for item in batch])
    # feature = torch.vstack([item[1] for item in batch])
    feature = torch.cat([item[1] for item in batch], dim=0)
    if label_type == "float":
        label = torch.FloatTensor([item[2] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[2] for item in batch])
    return [idx, feature, label]


def collate_features_mask(batch, label_type: str = "int"):
    idx = torch.LongTensor([item[0] for item in batch])
    feature = torch.cat([item[1] for item in batch], dim=0)
    if label_type == "float":
        label = torch.FloatTensor([item[2] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[2] for item in batch])
    mask = torch.cat([item[3] for item in batch], dim=0)
    return [idx, feature, label, mask]


def collate_ordinal_features(batch):
    idx = torch.LongTensor([item[0] for item in batch])
    feature = torch.cat([item[1] for item in batch], dim=0)
    label = torch.FloatTensor(np.array([item[2] for item in batch]))
    return [idx, feature, label]


def collate_survival_features(
    batch, label_type: str = "int", agg_method: str = "concat"
):
    idx = torch.LongTensor([item[0] for item in batch])
    # feature = torch.vstack([item[1] for item in batch])
    if agg_method == "concat" or not agg_method:
        feature = torch.cat([item[1] for item in batch], dim=0)
    elif agg_method == "self_att":
        feature = [item[1] for item in batch]
    if label_type == "float":
        label = torch.FloatTensor([item[2] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[2] for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    censorship = torch.FloatTensor([item[4] for item in batch])
    return [idx, feature, label, event_time, censorship]


def collate_survival_features_coords(
    batch, label_type: str = "int", agg_method: str = "concat"
):
    idx = torch.LongTensor([item[0] for item in batch])
    if agg_method == "concat":
        feature = torch.cat([item[1] for item in batch], dim=0)
        coords = torch.LongTensor(np.array([item[2] for item in batch]))
    elif agg_method == "self_att":
        feature = [item[1] for item in batch]
        coords = [
            [torch.LongTensor(item[2][i]) for i in range(len(item[2]))]
            for item in batch
        ]
    if label_type == "float":
        label = torch.FloatTensor([item[3] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[3] for item in batch])
    event_time = torch.FloatTensor([item[4] for item in batch])
    censorship = torch.FloatTensor([item[5] for item in batch])
    return [idx, feature, coords, label, event_time, censorship]


def collate_features_and_num_regions(batch, label_type: str = "int"):
    idx = torch.LongTensor([item[0] for item in batch])
    # feature = torch.vstack([item[1] for item in batch])
    feature = torch.cat([item[1] for item in batch], dim=0)
    num_regions = torch.LongTensor([item[2] for item in batch])
    if label_type == "float":
        label = torch.FloatTensor([item[3] for item in batch])
    elif label_type == "int":
        label = torch.LongTensor([item[3] for item in batch])
    return [idx, feature, num_regions, label]


def collate_region_filepaths(batch):
    item = batch[0]
    idx = torch.LongTensor([item[0]])
    fp = item[1]
    sid = item[2]
    pct = item[3]
    return [idx, fp, sid, pct]


def get_roc_auc_curve(
    probs: np.array(float), labels: List[int], log_to_wandb: bool = False
):
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    auc = metrics.roc_auc_score(labels, probs)
    fig = plt.figure(dpi=600)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.legend(loc="lower right")
    if log_to_wandb:
        img = wandb.Image(fig)
    plt.close()
    return fig


def plot_confusion_matrix(
    y_true: Union[List[float], np.array],
    y_pred: Union[List[float], np.array],
    show_pct: bool = False,
    cbar: bool = False,
    names: Optional[str] = None,
    normalize: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    Computes & plots confusion matrix

    Args:
        y_true (_type_): array-like of shape (n_samples,)
            Ground truth (correct) target values.
        y_pred (_type_): array-like of shape (n_samples,)
            Estimated targets as returned by a classifier.
        show_pct (bool, optional): _description_. Defaults to False.
        cbar (bool, optional): _description_. Defaults to False.
        names (Optional[str], optional): _description_. Defaults to None.
        normalize (Optional[str], optional): _description_. Defaults to None.
        title (Optional[str], optional): _description_. Defaults to None.
        save_path (Optional[str], optional): _description_. Defaults to None.
        dpi (int, optional): _description_. Defaults to 150.
    """

    cm = metrics.confusion_matrix(y_true, y_pred, normalize=normalize)
    cm_unnorm = metrics.confusion_matrix(y_true, y_pred, normalize=None)

    if not normalize:
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        annot2 = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i][0]
                    if show_pct:
                        annot[i, j] = f"{c}/{s}"
                        annot2[i, j] = f"\n\n{p:.2f}%"
                    else:
                        annot[i, j] = f"{c}/{s}"
                elif c == 0:
                    annot[i, j] = f"{c}"
                    annot2[i, j] = ""
                else:
                    if show_pct:
                        annot[i, j] = f"{c}"
                        annot2[i, j] = f"\n\n{p:.2f}%"
                    else:
                        annot[i, j] = f"{c}"

    else:
        cm_sum = np.sum(cm_unnorm, axis=1, keepdims=True)
        annot = np.empty_like(cm).astype(str)
        annot2 = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm_unnorm[i, j]
                p = cm[i, j] * 100
                if i == j:
                    s = cm_sum[i][0]
                    if show_pct:
                        annot[i, j] = f"{c}/{s}"
                        annot2[i, j] = f"\n\n{p:.1f}%"
                    else:
                        annot[i, j] = f"{c}/{s}"
                elif c == 0:
                    annot[i, j] = f"{c}"
                    annot2[i, j] = ""
                else:
                    if show_pct:
                        annot[i, j] = f"{c}"
                        annot2[i, j] = f"\n\n{p:.1f}%"
                    else:
                        annot[i, j] = f"{c}"

    if names and len(names) == cm.shape[0]:
        labels = [f"{str(n).upper()}" for n in names]
    else:
        labels = [f"{i}" for i in range(cm.shape[0])]

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig, ax = plt.subplots(dpi=dpi)

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        ax=ax,
        cmap="Blues",
        cbar=cbar,
        annot_kws={"size": "small"},
    )

    # Create a colormap with fully transparent colors
    cmap = sns.color_palette("Blues", as_cmap=True)
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:, -1] = 0.0
    transparent_cmap = matplotlib.colors.ListedColormap(cmap_colors)
    sns.heatmap(
        cm,
        annot=annot2,
        fmt="",
        cmap=transparent_cmap,
        cbar=False,
        annot_kws={"size": "xx-small"},
    )

    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("Groundtruth", labelpad=10)
    if title is not None:
        ax.set_title(title, pad=10)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def custom_isup_grade_dist(x: int, y: int):
    if (x == 0 and y == 1) or (x == 1 and y == 0):
        return 1.5
    else:
        return abs(x - y)


def get_majority_vote(
    preds, distance_func: Optional[Callable] = None, nfold: int = 5, seed: int = 0
):
    random.seed(seed)
    x = Counter(preds)
    max_occ = x.most_common(1)[0][1]
    ties = [p for p, occ in x.items() if occ == max_occ]
    if len(ties) == nfold:
        # [0,1,2,3,4] situation in 5-fold cv
        i = random.randint(0, len(ties) - 1)
        maj_vote = ties[i]
    elif len(ties) > 1:
        # [0,0,2,3,3] situation in 5-fold cv
        # in that case, break the tie by taking prediction closest to outlier
        outlier = [p for p, occ in x.items() if occ != max_occ][0]
        if distance_func:
            distances = [distance_func(t, outlier) for t in ties]
        else:
            distances = [abs(t - outlier) for t in ties]
        m = min(distances)
        idx_min = [i for i, v in enumerate(distances) if v == m]
        idx = random.randint(0, len(idx_min) - 1)
        i = idx_min[idx]
        maj_vote = ties[i]
    else:
        maj_vote = ties[0]
    return maj_vote


def update_log_dict(
    prefix,
    results,
    log_dict,
    step: Optional[str] = "step",
    to_log: Optional[List[str]] = None,
):
    if not to_log:
        to_log = list(results.keys())
    for r, v in results.items():
        if r in to_log:
            wandb.define_metric(f"{prefix}/{r}", step_metric=step)
            log_dict.update({f"{prefix}/{r}": v})


def make_weights_for_balanced_classes(dataset):
    n_samples = len(dataset)
    weight_per_class = []
    for c in range(dataset.num_classes):
        w = n_samples * 1.0 / len(dataset.class_2_id[c])
        weight_per_class.append(w)
    weight = []
    for idx in range(len(dataset)):
        y = dataset.get_label(idx)
        weight.append(weight_per_class[y])
    return torch.DoubleTensor(weight)


def get_preds_from_ordinal_logits(logits, loss: str):
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        if loss == "corn":
            prob = torch.cumprod(prob, dim=1)
    return prob, (prob > 0.5).cumprod(axis=1).sum(axis=1)


def get_label_from_ordinal_label(label):
    return (label > 0.5).cumprod(axis=1).sum(axis=1)


def get_label_from_regression_logits(logits, num_classes):
    pred = torch.max(
        torch.min(torch.round(logits), torch.Tensor([num_classes - 1])),
        torch.Tensor([0]),
    )
    # prob = (pred == logits) * torch.Tensor([1.0]) + (pred < logits) * (1 -(logits) % 1) + (pred > logits) * (logits % 1)
    # need to deal with border values, e.g. logits < -0.5 or logits > num_classes-0.5
    return pred


def aggregated_cindex(df: pd.DataFrame, label_name: str = "label", agg: str = "mean"):
    censorships = df.groupby("case_id").censorship.first()
    event_times = df.groupby("case_id")[label_name].first()
    if agg == "mean":
        risk_scores = df.groupby("case_id").risk.mean()
    elif agg == "max":
        risk_scores = df.groupby("case_id").risk.max()
    else:
        raise ValueError(f"agg ({agg}) argument not supported")
    c_index = concordance_index_censored(
        [bool(1 - c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]
    return c_index


def get_cumulative_dynamic_auc(
    train_df, test_df, risks, label_name, verbose: bool = False
):
    cols = ["censorship", label_name]
    train_tuples = train_df[cols].values
    tune_tuples = test_df[cols].values
    survival_train = np.array(
        list(zip(train_tuples[:, 0], train_tuples[:, 1])), dtype=np.dtype("bool,float")
    )
    survival_tune = np.array(
        list(zip(tune_tuples[:, 0], tune_tuples[:, 1])), dtype=np.dtype("bool,float")
    )
    train_min, train_max = train_df[label_name].min(), train_df[label_name].max()
    test_min, test_max = test_df[label_name].min(), test_df[label_name].max()
    min_y = math.ceil(test_min / 12)
    max_y = math.floor(test_max / 12)
    times = np.arange(min_y, max_y, 1)
    if train_min <= test_min < test_max < train_max:
        auc, mean_auc = cumulative_dynamic_auc(
            survival_train, survival_tune, risks, times * 12
        )
    else:
        if verbose:
            print(
                f"test data ({test_min},{test_max}) is not within time range of training data ({train_min},{train_max})"
            )
        auc, mean_auc = None, None
    return auc, mean_auc, times


def plot_cumulative_dynamic_auc(auc, mean_auc, times, epoch):
    fig = plt.figure(dpi=200)
    plt.plot(times, auc, marker="o")
    plt.axhline(mean_auc, linestyle="--")
    plt.xticks(times, [f"{int(t)}" for t in times])
    plt.xlabel("years from enrollment")
    plt.ylabel("time-dependent AUC")
    plt.title(f"Epoch {epoch+1}")
    plt.grid(True)
    return fig


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
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            self.optimizer = optim.SGD(
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
        params: Optional[dict] = None,
    ):
        self.scheduler = None
        self.name = params.name
        if self.name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=params.step_size, gamma=params.gamma
            )
        elif self.name == "cosine":
            assert (
                params.T_max != -1
            ), "T_max parameter must be specified! If you dont know what to use, plug in nepochs"
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                params.T_max, eta_min=params.eta_min
            )
        elif self.name == "reduce_lr_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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
        checkpoint_dir: Optional[Path] = None,
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


def train(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
    gradient_accumulation: Optional[int] = None,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
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
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):
            # optimizer.zero_grad()
            idx, x, label = batch
            x, label = x.to(device, non_blocking=True), label.to(
                device, non_blocking=True
            )
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

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(preds, labels, probs[:, 1])
        # roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        # results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(
            preds,
            labels,
            probs,
            class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
            use_wandb=use_wandb,
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
    collate_fn: Callable = partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)
                loss = criterion(logits, label)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(preds, labels, probs[:, 1])
        # roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        # results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(
            preds,
            labels,
            probs,
            class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
            use_wandb=use_wandb,
        )

    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Callable = partial(collate_features, label_type="int"),
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        desc=(f"Test"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(preds, labels, probs[:, 1])
        # roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        # results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(
            preds,
            labels,
            probs,
            class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
            use_wandb=use_wandb,
        )

    results.update(metrics)

    return results


def train_regression(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features, label_type="float"),
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
    gradient_accumulation: Optional[int] = None,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
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
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):
            # optimizer.zero_grad()
            idx, x, label = batch
            x, label = x.to(device, non_blocking=True), label.to(
                device, non_blocking=True
            )
            logits = model(x)
            loss = criterion(logits, label.unsqueeze(1))

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

            pred = get_label_from_regression_logits(logits.cpu(), dataset.num_classes)
            preds.extend(pred[:, 0].clone().tolist())

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune_regression(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features, label_type="float"),
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss = 0
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
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)
                loss = criterion(logits, label.unsqueeze(1))

                pred = get_label_from_regression_logits(
                    logits.cpu(), dataset.num_classes
                )
                preds.extend(pred[:, 0].clone().tolist())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()

    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test_regression(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Callable = partial(collate_features, label_type="float"),
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
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
        desc=(f"Test"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)

                pred = get_label_from_regression_logits(
                    logits.cpu(), dataset.num_classes
                )
                preds.extend(pred[:, 0].clone().tolist())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    return results


def train_regression_masked(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features_mask, label_type="float"),
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
    gradient_accumulation: Optional[int] = None,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
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
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):
            # optimizer.zero_grad()
            idx, x, label, pct = batch
            x, label, pct = (
                x.to(device, non_blocking=True),
                label.to(device, non_blocking=True),
                pct.to(device, non_blocking=True),
            )
            logits = model(x, pct=pct)
            loss = criterion(logits, label.unsqueeze(1))

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

            pred = get_label_from_regression_logits(logits.cpu(), dataset.num_classes)
            preds.extend(pred[:, 0].clone().tolist())

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune_regression_masked(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features_mask, label_type="float"),
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss = 0
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
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, label, pct = batch
                x, label, pct = (
                    x.to(device, non_blocking=True),
                    label.to(device, non_blocking=True),
                    pct.to(device, non_blocking=True),
                )
                logits = model(x, pct=pct)
                loss = criterion(logits, label.unsqueeze(1))

                pred = get_label_from_regression_logits(
                    logits.cpu(), dataset.num_classes
                )
                preds.extend(pred[:, 0].clone().tolist())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()

    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test_regression_masked(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Callable = partial(collate_features_mask, label_type="float"),
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
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
        desc=(f"Test"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, label, pct = batch
                x, label, pct = (
                    x.to(device, non_blocking=True),
                    label.to(device, non_blocking=True),
                    pct.to(device, non_blocking=True),
                )
                logits = model(x, pct=pct)

                pred = get_label_from_regression_logits(
                    logits.cpu(), dataset.num_classes
                )
                preds.extend(pred[:, 0].clone().tolist())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    return results


def train_ordinal(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    loss: str,
    collate_fn: Callable = collate_ordinal_features,
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
    gradient_accumulation: Optional[int] = None,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
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
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):
            # optimizer.zero_grad()
            idx, x, label = batch
            x, label = x.to(device, non_blocking=True), label.to(
                device, non_blocking=True
            )
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

            prob, pred = get_preds_from_ordinal_logits(logits, loss)
            preds.extend(pred.clone().tolist())

            label = get_label_from_ordinal_label(label)
            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune_ordinal(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    loss: str,
    collate_fn: Callable = collate_ordinal_features,
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss = 0
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
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)
                loss = criterion(logits, label)

                prob, pred = get_preds_from_ordinal_logits(logits, loss)
                preds.extend(pred.clone().tolist())

                label = get_label_from_ordinal_label(label)
                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

                epoch_loss += loss.item()

    dataset.df.loc[idxs, f"pred"] = preds

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test_ordinal(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    loss: str,
    collate_fn: Callable = collate_ordinal_features,
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    probs = np.empty((0, dataset.num_classes - 1))
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
        desc=(f"Test"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)

                prob, pred = get_preds_from_ordinal_logits(logits, loss)
                probs = np.append(probs, prob.cpu().detach().numpy(), axis=0)
                preds.extend(pred.clone().tolist())

                label = get_label_from_ordinal_label(label)
                labels.extend(label.clone().tolist())
                idxs.extend(idx.clone().tolist())

    dataset.df.loc[idxs, f"pred"] = preds
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()

    metrics = get_metrics(
        preds,
        labels,
        class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
        use_wandb=use_wandb,
    )
    results.update(metrics)

    return results


def train_survival(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    agg_method: Optional[str] = "concat",
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
    gradient_accumulation: Optional[int] = None,
    num_workers: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
    censorships, event_times = [], []
    risk_scores, labels = [], []
    idxs = []

    sampler = torch.utils.data.RandomSampler(dataset)
    if weighted_sampling:
        weights = make_weights_for_balanced_classes(dataset)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            len(weights),
        )

    if dataset.use_coords:
        collate_fn = partial(
            collate_survival_features_coords, label_type="int", agg_method=agg_method
        )
    else:
        collate_fn = partial(
            collate_survival_features, label_type="int", agg_method=agg_method
        )

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
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):
            if dataset.use_coords:
                idx, x, coords, label, event_time, c = batch
                if agg_method == "concat":
                    x, coords = x.to(device, non_blocking=True), coords.to(
                        device, non_blocking=True
                    )
                elif agg_method == "self_att":
                    x = [f.to(device, non_blocking=True) for f in x[0]]
                    coords = [c.to(device, non_blocking=True) for c in coords[0]]
            else:
                idx, x, label, event_time, c = batch
                if agg_method == "self_att":
                    x = [
                        xx[j].to(device, non_blocking=True)
                        for xx in x
                        for j in range(len(xx))
                    ]
                else:
                    x = x.to(device, non_blocking=True)
            label, c = label.to(device, non_blocking=True), c.to(
                device, non_blocking=True
            )

            if dataset.use_coords:
                logits = model(x, coords)  # [1, nbins]
            else:
                logits = model(x)  # [1, nbins]

            hazards = torch.sigmoid(logits)  # [1, nbins]
            surv = torch.cumprod(1 - hazards, dim=1)  # [1, nbins]

            loss = criterion(hazards, surv, label, c)

            loss_value = loss.item()
            epoch_loss += loss_value

            risk = -torch.sum(surv, dim=1).detach()  # [1]
            risk_scores.append(risk.item())
            censorships.append(c.item())
            event_times.append(event_time.item())

            if gradient_accumulation:
                loss = loss / gradient_accumulation

            loss.backward()

            if not gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()
            elif (i + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            labels.extend(label.clone().tolist())
            idxs.extend(list(idx))

    dataset.df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1 - c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune_survival(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    agg_method: Optional[str] = "concat",
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss = 0
    censorships, event_times = [], []
    risk_scores, labels = [], []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)
    if dataset.use_coords:
        collate_fn = partial(
            collate_survival_features_coords, label_type="int", agg_method=agg_method
        )
    else:
        collate_fn = partial(
            collate_survival_features, label_type="int", agg_method=agg_method
        )

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
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                if dataset.use_coords:
                    idx, x, coords, label, event_time, c = batch
                    if agg_method == "concat":
                        x, coords = x.to(device, non_blocking=True), coords.to(
                            device, non_blocking=True
                        )
                    elif agg_method == "self_att":
                        x = [f.to(device, non_blocking=True) for f in x[0]]
                        coords = [c.to(device, non_blocking=True) for c in coords[0]]
                else:
                    idx, x, label, event_time, c = batch
                    if agg_method == "self_att":
                        x = [
                            xx[j].to(device, non_blocking=True)
                            for xx in x
                            for j in range(len(xx))
                        ]
                    else:
                        x = x.to(device, non_blocking=True)
                label, c = label.to(device, non_blocking=True), c.to(
                    device, non_blocking=True
                )

                if dataset.use_coords:
                    logits = model(x, coords)
                else:
                    logits = model(x)

                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1 - hazards, dim=1)

                loss = criterion(hazards, surv, label, c, alpha=0)
                epoch_loss += loss.item()

                risk = -torch.sum(surv, dim=1).detach()
                risk_scores.append(risk.item())
                censorships.append(c.item())
                event_times.append(event_time.item())

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))

    dataset.df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1 - c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index
    results["risks"] = risk_scores

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test_survival(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    agg_method: Optional[str] = "concat",
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    censorships, event_times = [], []
    risk_scores = []
    idxs = []

    sampler = torch.utils.data.SequentialSampler(dataset)
    if dataset.use_coords:
        collate_fn = partial(
            collate_survival_features_coords, label_type="int", agg_method=agg_method
        )
    else:
        collate_fn = partial(
            collate_survival_features, label_type="int", agg_method=agg_method
        )

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
        desc=(f"Test"),
        unit=" patient",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                if dataset.use_coords:
                    idx, x, coords, _, event_time, c = batch
                    if agg_method == "concat":
                        x, coords = x.to(device, non_blocking=True), coords.to(
                            device, non_blocking=True
                        )
                    elif agg_method == "self_att":
                        x = [f.to(device, non_blocking=True) for f in x[0]]
                        coords = [c.to(device, non_blocking=True) for c in coords[0]]
                else:
                    idx, x, _, event_time, c = batch
                    if agg_method == "self_att":
                        x = [
                            xx[j].to(device, non_blocking=True)
                            for xx in x
                            for j in range(len(xx))
                        ]
                    else:
                        x = x.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)

                if dataset.use_coords:
                    logits = model(x, coords)
                else:
                    logits = model(x)

                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1 - hazards, dim=1)

                risk = -torch.sum(surv, dim=1).detach()
                risk_scores.append(risk.item())
                censorships.append(c.item())
                event_times.append(event_time.item())

                idxs.extend(list(idx))

    dataset.df.loc[idxs, "risk"] = risk_scores

    c_index = concordance_index_censored(
        [bool(1 - c) for c in censorships],
        event_times,
        risk_scores,
        tied_tol=1e-08,
    )[0]

    results["c-index"] = c_index

    return results


def train_on_regions(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features_and_num_regions, label_type="int"),
    batch_size: Optional[int] = 1,
    weighted_sampling: Optional[bool] = False,
    gradient_accumulation: Optional[int] = None,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs, num_regions = [], []

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
        num_workers=num_workers,
    )

    results = {}

    with tqdm.tqdm(
        loader,
        desc=(f"Epoch {epoch} - Train"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        for i, batch in enumerate(t):
            # optimizer.zero_grad()
            idx, x, M, label = batch
            x, label = x.to(device, non_blocking=True), label.to(
                device, non_blocking=True
            )
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
            num_regions.extend(list(M))

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds
    dataset.df.loc[idxs, f"num_regions_sampled"] = [int(m) for m in num_regions]

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(preds, labels, probs[:, 1])
        # roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        # results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(
            preds,
            labels,
            probs,
            class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
            use_wandb=use_wandb,
        )

    results.update(metrics)

    train_loss = epoch_loss / len(loader)
    results["loss"] = train_loss

    return results


def tune_on_regions(
    epoch: int,
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    criterion: Callable,
    collate_fn: Callable = partial(collate_features_and_num_regions, label_type="int"),
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss = 0
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs, num_regions = [], []

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
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=False,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, M, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)
                loss = criterion(logits, label)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))
                num_regions.extend(list(M))

                epoch_loss += loss.item()

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds
    dataset.df.loc[idxs, f"num_regions_sampled"] = [int(m) for m in num_regions]

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(preds, labels, probs[:, 1])
        # roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        # results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(
            preds,
            labels,
            probs,
            class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
            use_wandb=use_wandb,
        )

    results.update(metrics)

    tune_loss = epoch_loss / len(loader)
    results["loss"] = tune_loss

    return results


def test_on_regions(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collate_fn: Callable = partial(collate_features_and_num_regions, label_type="int"),
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    probs = np.empty((0, dataset.num_classes))
    preds, labels = [], []
    idxs, num_regions = [], []

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
        desc=(f"Test"),
        unit=" slide",
        ncols=80,
        unit_scale=batch_size,
        leave=True,
    ) as t:
        with torch.no_grad():
            for i, batch in enumerate(t):
                idx, x, M, label = batch
                x, label = x.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )
                logits = model(x)

                pred = torch.topk(logits, 1, dim=1)[1]
                preds.extend(pred[:, 0].clone().tolist())

                prob = F.softmax(logits, dim=1).cpu().detach().numpy()
                probs = np.append(probs, prob, axis=0)

                labels.extend(label.clone().tolist())
                idxs.extend(list(idx))
                num_regions.extend(list(M))

    # TODO: what happens if idxs is not made of unique index values?
    for class_idx, p in enumerate(probs.T):
        dataset.df.loc[idxs, f"prob_{class_idx}"] = p.tolist()
    dataset.df.loc[idxs, f"pred"] = preds
    dataset.df.loc[idxs, f"num_regions_sampled"] = [int(m) for m in num_regions]

    if dataset.num_classes == 2:
        metrics = get_binary_metrics(preds, labels, probs[:, 1])
        # roc_auc_curve = get_roc_auc_curve(probs[:, 1], labels)
        # results.update({"roc_auc_curve": roc_auc_curve})
    else:
        metrics = get_metrics(
            preds,
            labels,
            probs,
            class_names=[f"isup_{i}" for i in range(dataset.num_classes)],
            use_wandb=use_wandb,
        )

    results.update(metrics)

    return results
