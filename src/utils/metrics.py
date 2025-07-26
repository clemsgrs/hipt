import numpy as np
from sklearn import metrics
from sksurv.metrics import concordance_index_censored


def get_metrics(
    names: list[str],
    preds: list[int],
    labels: list[int],
    probs: np.ndarray | None = None,
    multi_class: str = "ovr",
    event_indicator: list[bool] | None = None,
):
    metrics_dict = {}
    labels = np.asarray(labels)
    for metric_name in names:
        if metric_name == "auc":
            assert probs is not None, "AUC requires probabilities"
            if probs.shape[1] == 2:
                # if binary classification, take the second column for positive class
                probs = probs[:, 1]
            auc = metrics.roc_auc_score(labels, probs, multi_class=multi_class)
            metrics_dict.update({"auc": auc})
        if metric_name == "quadratic_kappa":
            kappa = metrics.cohen_kappa_score(labels, preds, weights="quadratic")
            metrics_dict.update({"quadratic_kappa": kappa})
        if metric_name == "c-index":
            assert event_indicator is not None, "c-index requires event indicators"
            c_index = concordance_index_censored(
                event_indicator,
                labels,
                preds,
                tied_tol=1e-08,
            )[0]
            metrics_dict.update({"c-index": c_index})
    return metrics_dict