# =========================
# Imports
# =========================

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

import numpy as np


import numpy as np

def to_native(obj):
    if isinstance(obj, np.generic):  # handles np.int64, np.float64, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj

def confusion_matrix_dict(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)

    return to_native({
        "labels": labels.tolist(),
        "matrix": cm.tolist(),
        "rows": [
            {
                "actual": labels[i],
                "values": cm[i].tolist()
            }
            for i in range(len(labels))
        ]
    })

CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
    "confusion_matrix": confusion_matrix_dict,
    "roc_auc": roc_auc_score,
    "log_loss": log_loss,
}


# =========================
# Regression Metrics
# =========================

REGRESSION_METRICS = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "r2": r2_score,
    "mape": mean_absolute_percentage_error,
}


METRICS = {**REGRESSION_METRICS,**CLASSIFICATION_METRICS}