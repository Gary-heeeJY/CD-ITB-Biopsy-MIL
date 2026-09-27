"""Evaluation helpers shared by training and external-test entry points."""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_curve


POSITIVE_CLASS_NAME = "CD"
POSITIVE_CLASS_INDEX = 1


def youden_threshold(labels, positive_scores, default=0.5):
    """Return a threshold fitted on validation labels and CD probabilities."""
    labels = np.asarray(labels).reshape(-1)
    positive_scores = np.asarray(positive_scores).reshape(-1)
    if labels.size == 0 or positive_scores.size == 0 or np.unique(labels).size < 2:
        return float(default)
    try:
        fpr, tpr, thresholds = roc_curve(labels, positive_scores, pos_label=POSITIVE_CLASS_INDEX)
        finite = np.isfinite(thresholds)
        if not finite.any():
            return float(default)
        scores = tpr - fpr
        scores[~finite] = -np.inf
        return float(thresholds[int(np.argmax(scores))])
    except (ValueError, IndexError):
        return float(default)


def checkpoint_threshold(checkpoint, default=0.5):
    """Read a validation-fitted threshold from a checkpoint."""
    if isinstance(checkpoint, dict):
        value = checkpoint.get("case_threshold")
        if value is None:
            metrics = checkpoint.get("metrics")
            if isinstance(metrics, dict):
                value = metrics.get("threshold")
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return float(default)


def save_validation_predictions(output_dir, labels, positive_scores):
    """Save one fold's best validation predictions for pooled calibration."""
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "validation_predictions.npz"),
        labels=np.asarray(labels, dtype=np.int64),
        positive_scores=np.asarray(positive_scores, dtype=np.float64),
    )


def pooled_validation_threshold(experiment_dir, folds=5, default=0.5):
    """Compute one Youden threshold from all saved fold-validation predictions."""
    labels = []
    scores = []
    for fold_idx in range(int(folds)):
        path = os.path.join(experiment_dir, f"fold-{fold_idx}", "validation_predictions.npz")
        if not os.path.isfile(path):
            continue
        data = np.load(path)
        labels.append(np.asarray(data["labels"]))
        scores.append(np.asarray(data["positive_scores"]))
    if not labels:
        return float(default)
    return youden_threshold(np.concatenate(labels), np.concatenate(scores), default=default)


def write_pooled_validation_threshold(experiment_dir, folds=5, default=0.5):
    """Write the pooled validation threshold used by external test scripts."""
    os.makedirs(experiment_dir, exist_ok=True)
    threshold = pooled_validation_threshold(experiment_dir, folds=folds, default=default)
    payload = {
        "threshold": float(threshold),
        "selection": "pooled_out_of_fold_validation_youden",
        "folds": int(folds),
    }
    with open(os.path.join(experiment_dir, "validation_threshold.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return float(threshold)


def load_pooled_validation_threshold(experiment_dir, default=None):
    """Load a previously calibrated pooled threshold, if available."""
    path = os.path.join(experiment_dir, "validation_threshold.json")
    if not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle).get("threshold")
        return float(value) if value is not None and np.isfinite(float(value)) else default
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default
