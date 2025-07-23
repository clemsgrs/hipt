import numpy as np
from sklearn import metrics


def get_metrics(
    names: list[str],
    preds: list[int],
    labels: list[int],
    probs: np.ndarray | None = None,
    multi_class: str = "ovr",
):
    metrics_dict = {}
    labels = np.asarray(labels)
    for metric_name in names:
        if metric_name == "auc":
            assert probs is not None, "AUC requires probabilities"
            auc = metrics.roc_auc_score(labels, probs, multi_class=multi_class)
            metrics_dict.update({"auc": auc})
        if metric_name == "quadratic_kappa":
            kappa = metrics.cohen_kappa_score(labels, preds, weights="quadratic")
            metrics_dict.update({"quadratic_kappa": kappa})
    return metrics_dict
