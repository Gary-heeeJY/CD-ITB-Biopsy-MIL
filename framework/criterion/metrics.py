"""Metrics and feature collectors used by the reported CD-versus-ITB runs."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from criterion.metric_utils import (
    get_accuracy,
    get_best_case_threshold,
    get_case_AUC,
    get_precision,
    get_sensitivity,
    get_specificity,
)


def _result_path(cfg):
    return getattr(cfg, "logs_dir_", ".") if cfg is not None else "."


def _threshold(positive_scores, labels, phase):
    if phase in {"train", "eval", "val"}:
        return float(get_best_case_threshold(torch.as_tensor(positive_scores), torch.as_tensor(labels)))
    return 0.5


def _safe_auc(labels, scores, multi_class="ovr"):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        if scores.ndim == 1:
            return float(roc_auc_score(labels, scores))
        return float(roc_auc_score(labels, scores, average="macro", multi_class=multi_class))
    except ValueError:
        return 0.0


class predict_metric:
    """Collect slide-level softmax outputs and report multiclass metrics."""

    def __init__(self, best_metric="micro_AUC", cfg=None, phase="train"):
        self.best_metric = best_metric
        self.result_path = _result_path(cfg)
        self.value = 0.0
        self.reset(phase)

    def reset(self, phase="train"):
        self.probs = []
        self.labels = []
        self.slide_name = []
        self.phase = phase

    def update(self, prob, label, slide_id):
        if not isinstance(prob, torch.Tensor) or not isinstance(label, torch.Tensor):
            raise TypeError("prob and label must be torch.Tensor instances")
        probs = F.softmax(prob, dim=1).detach().cpu().numpy()
        labels = label.detach().cpu().reshape(-1).numpy().astype(int)
        names = list(slide_id) if isinstance(slide_id, (list, tuple)) else [slide_id] * len(labels)
        self.probs.extend(probs)
        self.labels.extend(labels.tolist())
        self.slide_name.extend(names[: len(labels)])

    def save_excel(self, epoch):
        os.makedirs(self.result_path, exist_ok=True)
        probs = np.asarray(self.probs)
        data = {"id": range(len(self.labels)), "slide_id": self.slide_name, "label": self.labels}
        for index in range(probs.shape[1]):
            data[f"prob_{index}"] = probs[:, index]
        pd.DataFrame(data).to_csv(os.path.join(self.result_path, f"first_prediction_slide_{epoch}.csv"), index=False)

    @property
    def get_value(self):
        return self.value

    def __str__(self):
        probs = np.asarray(self.probs)
        labels = np.asarray(self.labels)
        if len(labels) == 0:
            return "accuracy:0.0000  sensitivity:0.0000  precision:0.0000  AUC:0.0000  F1_score:0.0000"
        pred = probs.argmax(axis=1)
        auc = _safe_auc(labels, probs)
        self.value = auc
        return (
            f"accuracy:{accuracy_score(labels, pred):.4f}  "
            f"sensitivity:{recall_score(labels, pred, average='weighted', zero_division=0):.4f}  "
            f"precision:{precision_score(labels, pred, average='weighted', zero_division=0):.4f}  "
            f"AUC:{auc:.4f}  "
            f"F1_score:{f1_score(labels, pred, average='weighted', zero_division=0):.4f}"
        )


class predict_slide_metric(predict_metric):
    def update(self, prob, label):
        super().update(prob, label, [""] * len(label))


class Biospy_patient_metric:
    """Collect case-level binary outputs; CD is class 1 and the positive class."""

    def __init__(self, cfg=None, phase="train"):
        self.result_path = _result_path(cfg)
        self.value = 0.0
        self.reset(phase)

    def reset(self, phase="train"):
        self.probs = []
        self.labels = []
        self.case_name = []
        self.phase = phase

    def update(self, prob, label, case_name):
        if not isinstance(prob, torch.Tensor) or not isinstance(label, torch.Tensor):
            raise TypeError("prob and label must be torch.Tensor instances")
        probs = F.softmax(prob, dim=-1).detach().cpu().numpy().reshape(-1, prob.shape[-1])
        labels = label.detach().cpu().reshape(-1).numpy().astype(int)
        names = list(case_name) if isinstance(case_name, (list, tuple)) else [case_name] * len(labels)
        self.probs.extend(probs)
        self.labels.extend(labels.tolist())
        self.case_name.extend(names[: len(labels)])

    def save_excel(self, epoch):
        os.makedirs(self.result_path, exist_ok=True)
        probs = np.asarray(self.probs)
        pd.DataFrame({
            "id": range(len(self.labels)),
            "case_id": self.case_name,
            "prob_0": probs[:, 0],
            "prob_1": probs[:, 1],
            "label": self.labels,
        }).to_csv(os.path.join(self.result_path, f"prediction_case_{epoch}.csv"), index=False)

    @property
    def get_value(self):
        return self.value

    def __str__(self):
        probs = np.asarray(self.probs)
        labels = np.asarray(self.labels)
        if len(labels) == 0:
            return "accuracy:0.0000  sensitivity:0.0000  specificity:0.0000  precision:0.0000  AUC:0.0000  F1_score:0.0000  Threshold:0.5"
        positive = probs[:, 1]
        threshold = _threshold(positive, labels, self.phase)
        pred = (positive >= threshold).astype(int)
        accuracy = accuracy_score(labels, pred)
        sensitivity = recall_score(labels, pred, zero_division=0)
        specificity = recall_score(labels, pred, pos_label=0, zero_division=0)
        precision = precision_score(labels, pred, zero_division=0)
        f1 = f1_score(labels, pred, zero_division=0)
        auc = _safe_auc(labels, positive)
        self.value = auc
        return (
            f"accuracy:{accuracy:.4f}  sensitivity:{sensitivity:.4f}  "
            f"specificity:{specificity:.4f}  precision:{precision:.4f}  "
            f"AUC:{auc:.4f}  F1_score:{f1:.4f}  Threshold:{threshold:.4f}"
        )


class patient_feat:
    def __init__(self, cfg=None):
        self.reset()

    def reset(self, phase="train"):
        self.case_name = []
        self.case_info = {}
        self.phase = phase

    def update(self, case_name, slide_feat, case_label, logit, slide_label):
        if not isinstance(slide_feat, torch.Tensor) or not isinstance(case_label, torch.Tensor):
            raise TypeError("slide_feat and case_label must be torch.Tensor instances")
        probs = F.softmax(logit, dim=1)
        if case_name not in self.case_info:
            self.case_name.append(case_name)
            self.case_info[case_name] = {
                "case_feat": slide_feat,
                "case_label": case_label,
                "logit": probs,
                "slide_label": slide_label,
            }
        else:
            item = self.case_info[case_name]
            item["case_feat"] = torch.cat((item["case_feat"], slide_feat), dim=0)
            item["logit"] = torch.cat((item["logit"], probs), dim=0)
            item["slide_label"] = torch.cat((item["slide_label"], slide_label), dim=0)

    @property
    def get_value(self):
        return self.case_info

    def str(self):
        return self.case_info


class patient_feat_feedback:
    def __init__(self, cfg=None):
        self.reset()

    def reset(self, phase="train"):
        self.case_info = {}
        self.phase = phase

    def update(self, case_name, case_feat):
        self.case_info[case_name] = {"case_feat": case_feat}

    @property
    def get_value(self):
        return self.case_info

    def str(self):
        return self.case_info


class CAME_patient_metric(Biospy_patient_metric):
    pass


class CAME_patient_metric_ori(Biospy_patient_metric):
    pass


class Biospy_patient_metric_ori(Biospy_patient_metric):
    pass


def get_metric(cfg):
    return predict_metric(cfg=cfg)


def get_slide_metric(cfg):
    return predict_slide_metric(cfg=cfg)


def get_patient_feat(cfg):
    return patient_feat(cfg=cfg)


def get_patient_feat_feedback(cfg):
    return patient_feat_feedback(cfg=cfg)


def get_came_patient_metric(cfg):
    return CAME_patient_metric(cfg=cfg)


def get_came_patient_metric_ori(cfg):
    return CAME_patient_metric_ori(cfg=cfg)


def get_biospy_patient_metric(cfg):
    return Biospy_patient_metric(cfg=cfg)


def get_biospy_patient_metric_ori(cfg):
    return Biospy_patient_metric_ori(cfg=cfg)
