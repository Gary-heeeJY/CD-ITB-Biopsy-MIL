"""
Evaluate direct patch-to-patient MIL baselines on an independent test set.

The tested baselines are ABMIL, TransMIL, and MambaMIL trained by
scripts/train_patient_mil_baselines.py. Each model maps all patch features from a
patient directly to the patient label. Fold predictions are averaged for the
final ensemble result.
"""

import argparse
import os
import sys
import warnings

current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]
sys.path.append(root_path)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, SequentialSampler

from config.config import _C as base_cfg
from data.patient_mil_dataset import PatientMILBagDataset, patient_collate_mil
from models import build_model
from utils_ import mkdir, setup_logger
from utils_.evaluation import checkpoint_threshold, load_pooled_validation_threshold

warnings.filterwarnings("ignore")


BASELINE_CONFIGS = {
    "ABMIL": {
        "cfg": "./configs/biospy_case_abmil.yaml",
        "model_type": "GatedAttention",
    },
    "TransMIL": {
        "cfg": "./configs/TransMIL.yaml",
        "model_type": "TransMIL",
    },
    "MambaMIL": {
        "cfg": "./configs/MambaMIL.yaml",
        "model_type": "MambaMIL",
    },
}


CHECKPOINT_CANDIDATES = (
    "patient_auc_best_model.pt",
    "patient_best_model.pt",
    "last_model.pt",
)


def _parse_csv_list(text):
    if text is None or text == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _configure(cfg_path, model_name, model_type, args):
    cfg = base_cfg.clone()
    cfg.merge_from_file(cfg_path)
    cfg.defrost()
    cfg.model_type = model_type
    cfg.Task_name = f"{args.task_prefix}_{model_name}_Patch2Patient"
    cfg.seed = args.seed
    cfg.DATA.batch_size = 1
    cfg.DATA.n_classes = cfg.DATA.case_classes
    cfg.DATA.label_csv_file = args.label_csv_file
    cfg.DATA.pt_path = args.feat_file
    if args.gpu_id is not None:
        cfg.gpu_id = args.gpu_id
    if args.num_workers is not None:
        cfg.DATA.num_workers = args.num_workers
    cfg.freeze()
    return cfg


def _model_logits(output):
    if isinstance(output, dict):
        logits = output["logits"]
    elif isinstance(output, (tuple, list)):
        logits = output[0]
    else:
        logits = output
    if logits.dim() == 3:
        logits = logits.squeeze(1)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return logits


def _build_test_loader(cfg, args):
    label_df = pd.read_csv(args.label_csv_file)
    split_ids = label_df["slide_id"].dropna().astype(str).tolist()
    dataset = PatientMILBagDataset(
        cfg,
        split_ids=split_ids,
        phase="test",
        max_patches_per_patient=args.max_patches_per_patient,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        num_workers=getattr(cfg.DATA, "num_workers", 2),
        collate_fn=patient_collate_mil,
    )


def _checkpoint_dir(args, cfg, fold_idx, model_name):
    if args.checkpoint_template:
        return args.checkpoint_template.format(
            checkpoint_root=args.checkpoint_root,
            task_name=cfg.Task_name,
            seed=cfg.seed,
            fold_idx=fold_idx,
            model_name=model_name,
        )
    return os.path.join(args.checkpoint_root, cfg.Task_name, f"seed-{cfg.seed}", f"fold-{fold_idx}")


def _find_checkpoint(args, cfg, fold_idx, model_name):
    fold_dir = _checkpoint_dir(args, cfg, fold_idx, model_name)
    for filename in CHECKPOINT_CANDIDATES:
        path = os.path.join(fold_dir, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {fold_dir}. Tried: {CHECKPOINT_CANDIDATES}")


def _build_model(cfg, device):
    dummy_case_model = nn.Identity()
    model, _ = build_model(cfg, dummy_case_model)
    model.to(device)
    return model


def _load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def _predict_fold(model, loader, device, threshold):
    model.eval()
    rows = []
    with torch.no_grad():
        for feat, label, case_id, slide_ids in loader:
            feat = feat.to(device)
            logits = _model_logits(model(feat))
            prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(prob[1] >= threshold)
            rows.append(
                {
                    "CaseID": case_id,
                    "Label": int(label.item()),
                    "Prob_CD": float(prob[1]),
                    "Prob_TB": float(prob[0]),
                    "Pred": pred,
                    "NumSlides": len(slide_ids),
                }
            )
    return rows


def _metric_summary(y_true, y_prob_pos, y_pred, threshold):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": None,
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) > 1:
        result["auc"] = roc_auc_score(y_true, y_prob_pos)
    return result


def _log_metrics(logger, title, y_true, y_prob_pos, y_pred, threshold):
    summary = _metric_summary(y_true, y_prob_pos, y_pred, threshold)
    logger.info("--- %s ---", title)
    logger.info("Accuracy: %.4f", summary["accuracy"])
    logger.info("F1-macro: %.4f", summary["f1_macro"])
    if summary["auc"] is not None:
        logger.info("AUC: %.4f", summary["auc"])
    logger.info("Threshold: %.4f", summary["threshold"])
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_true, y_pred, labels=[0, 1]))
    logger.info(
        "Report:\n%s",
        classification_report(y_true, y_pred, labels=[0, 1], target_names=["ITB", "CD"], digits=4, zero_division=0),
    )
    return summary


def _evaluate_one_model(model_name, cfg_path, model_type, args, device):
    cfg = _configure(cfg_path, model_name, model_type, args)
    out_dir = os.path.join(args.ex_result, model_name)
    mkdir(out_dir)
    logger = setup_logger(f"test_patient_mil_{model_name}", os.path.join(out_dir, "test_patient_mil_log.txt"))
    logger.info("Testing %s", model_name)
    logger.info("Using Config:\n%s", cfg)

    loader = _build_test_loader(cfg, args)
    ensemble = {}
    validation_thresholds = []
    pooled_threshold = load_pooled_validation_threshold(
        os.path.join(args.checkpoint_root, cfg.Task_name, f"seed-{cfg.seed}")
    )
    fold_start = args.fold_start
    fold_end = args.fold_end if args.fold_end is not None else args.folds

    for fold_idx in range(fold_start, fold_end):
        logger.info("Processing fold %d", fold_idx)
        model = _build_model(cfg, device)
        checkpoint_path = _find_checkpoint(args, cfg, fold_idx, model_name)
        logger.info("Loading checkpoint: %s", checkpoint_path)
        checkpoint = _load_model(model, checkpoint_path, device)
        validation_thresholds.append(checkpoint_threshold(checkpoint))

        fold_threshold = pooled_threshold if pooled_threshold is not None else validation_thresholds[-1]
        fold_rows = _predict_fold(model, loader, device, fold_threshold)
        pd.DataFrame(fold_rows).to_csv(os.path.join(out_dir, f"fold_{fold_idx}_predictions.csv"), index=False)

        y_true = np.array([row["Label"] for row in fold_rows])
        y_prob_pos = np.array([row["Prob_CD"] for row in fold_rows])
        y_pred = np.array([row["Pred"] for row in fold_rows])
        _log_metrics(logger, f"Fold {fold_idx} Metrics", y_true, y_prob_pos, y_pred, fold_threshold)

        for row in fold_rows:
            case_id = row["CaseID"]
            prob = np.array([row["Prob_TB"], row["Prob_CD"]], dtype=np.float64)
            if case_id not in ensemble:
                ensemble[case_id] = {
                    "prob_sum": np.zeros_like(prob),
                    "label": row["Label"],
                    "count": 0,
                    "num_slides": row["NumSlides"],
                }
            ensemble[case_id]["prob_sum"] += prob
            ensemble[case_id]["count"] += 1

    final_rows = []
    y_true, y_prob_pos = [], []
    for case_id, item in ensemble.items():
        avg_prob = item["prob_sum"] / max(item["count"], 1)
        y_true.append(item["label"])
        y_prob_pos.append(avg_prob[1])
        final_rows.append(
            {
                "CaseID": case_id,
                "Label": item["label"],
                "Avg_Prob_CD": float(avg_prob[1]),
                "Avg_Prob_TB": float(avg_prob[0]),
                "Ensemble_Pred": int(np.argmax(avg_prob)),
                "Fold_Count": item["count"],
                "NumSlides": item["num_slides"],
            }
        )

    y_true = np.array(y_true)
    y_prob_pos = np.array(y_prob_pos)
    threshold = (
        float(pooled_threshold)
        if pooled_threshold is not None
        else (float(np.mean(validation_thresholds)) if validation_thresholds else 0.5)
    )
    y_pred = (y_prob_pos > threshold).astype(int)
    for idx, row in enumerate(final_rows):
        row["Threshold_Pred"] = int(y_pred[idx])

    pd.DataFrame(final_rows).to_csv(os.path.join(out_dir, "Patient_Final_Results.csv"), index=False)
    summary = _log_metrics(logger, "Fold Ensemble Metrics", y_true, y_prob_pos, y_pred, threshold)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "Patient_Final_Metrics.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate patch-to-patient MIL baselines on an independent test set")
    parser.add_argument("--models", default="ABMIL,TransMIL,MambaMIL", help="Comma-separated subset to test.")
    parser.add_argument("--abmil-cfg", default=BASELINE_CONFIGS["ABMIL"]["cfg"], type=str)
    parser.add_argument("--transmil-cfg", default=BASELINE_CONFIGS["TransMIL"]["cfg"], type=str)
    parser.add_argument("--mamba-cfg", default=BASELINE_CONFIGS["MambaMIL"]["cfg"], type=str)
    parser.add_argument("--label-csv-file", default="./data/labels/external_test.csv", type=str)
    parser.add_argument("--feat-file", default="./data/features/MUSK_CORAL", type=str)
    parser.add_argument("--checkpoint-root", default="./outputs/AJP_Comparison_experiments", type=str)
    parser.add_argument("--checkpoint-template", default=None, type=str)
    parser.add_argument("--ex-result", default="./outputs/AJP_Comparison_experiments/MUSK_DirectMIL_External_Test", type=str)
    parser.add_argument("--task-prefix", default="MUSK_DirectMIL", type=str)
    parser.add_argument("--seed", default=421, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--fold-start", default=0, type=int)
    parser.add_argument("--fold-end", default=None, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--max-patches-per-patient", default=0, type=int)
    parser.add_argument("--cuda-visible-devices", default=None, type=str)
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    mkdir(args.ex_result)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    cfg_paths = {
        "ABMIL": args.abmil_cfg,
        "TransMIL": args.transmil_cfg,
        "MambaMIL": args.mamba_cfg,
    }

    for model_name in _parse_csv_list(args.models):
        if model_name not in BASELINE_CONFIGS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {list(BASELINE_CONFIGS)}")
        _evaluate_one_model(
            model_name=model_name,
            cfg_path=cfg_paths[model_name],
            model_type=BASELINE_CONFIGS[model_name]["model_type"],
            args=args,
            device=device,
        )


if __name__ == "__main__":
    main()

# End of baseline evaluation entry point.
