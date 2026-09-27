"""
Run direct patch-to-patient MIL baselines in one command.

This script is for the first comparison group in the paper:
ABMIL, TransMIL, and MambaMIL aggregate all patch features from a patient
directly to the patient label, without the proposed slide-to-patient hierarchy.
"""

import argparse
import os
import sys
import warnings

current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn

from config.config import _C as base_cfg
from data.patient_mil_dataset import build_patient_mil_loaders
from engine.train_val_patient_mil import train_val_patient_mil
from models import build_model
from optimizer import build_lr_scheduler
from utils_ import mkdir, seed_torch, setup_logger
from utils_.evaluation import write_pooled_validation_threshold

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


def _parse_csv_list(text):
    if text is None or text == "":
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def _configure(cfg_path, model_name, model_type, args):
    cfg = base_cfg.clone()
    cfg.merge_from_file(cfg_path)
    cfg.defrost()
    cfg.model_type = model_type
    cfg.Task_name = f"{args.task_prefix}_{model_name}_Patch2Patient"
    cfg.DATA.batch_size = 1
    cfg.DATA.n_classes = cfg.DATA.case_classes
    cfg.seed = args.seed
    if args.gpu_id is not None:
        cfg.gpu_id = args.gpu_id
    if args.num_epochs is not None:
        cfg.SOLVER.num_epochs = args.num_epochs
    if args.fold_k is not None:
        cfg.DATA.fold_K = args.fold_k
    if args.num_workers is not None:
        cfg.DATA.num_workers = args.num_workers
    if args.result_root is not None:
        cfg.logs_dir = args.result_root
        cfg.model_dir = args.result_root
    cfg.freeze()
    return cfg


def _build_weighted_patient_loss(cfg, train_num_list):
    counts = torch.tensor(train_num_list, dtype=torch.float32)
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    return nn.CrossEntropyLoss(weight=weights.to(device))


def _run_one_baseline(model_name, cfg_path, model_type, args):
    cfg = _configure(cfg_path, model_name, model_type, args)
    seed_torch(seed=cfg.seed)

    fold_start = args.fold_start if args.fold_start is not None else cfg.DATA.fold_idx
    fold_end = args.fold_end if args.fold_end is not None else cfg.DATA.fold_K

    for fold_idx in range(fold_start, fold_end):
        cfg.defrost()
        cfg.DATA.fold_idx = fold_idx
        cfg.logs_dir_ = os.path.join(cfg.logs_dir, cfg.Task_name, f"seed-{cfg.seed}", f"fold-{fold_idx}")
        cfg.model_dir_ = os.path.join(cfg.model_dir, cfg.Task_name, f"seed-{cfg.seed}", f"fold-{fold_idx}")
        cfg.freeze()

        mkdir(cfg.logs_dir_)
        mkdir(cfg.model_dir_)
        setup_logger("eval", os.path.join(cfg.logs_dir_, "eval.log"))
        logger = setup_logger("train", os.path.join(cfg.logs_dir_, "train.log"))
        logger.info("Using Config:\n%s", cfg)

        loaders = build_patient_mil_loaders(
            cfg,
            max_patches_per_patient=args.max_patches_per_patient,
        )
        criterion = _build_weighted_patient_loss(cfg, loaders["train_num_list"])

        dummy_case_model = nn.Identity()
        model, _ = build_model(cfg, dummy_case_model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.SOLVER.lr,
            betas=(cfg.SOLVER.beta1, cfg.SOLVER.beta2),
            weight_decay=cfg.SOLVER.weight_decay,
        )
        lr_scheduler, optimizer = build_lr_scheduler(cfg, optimizer)

        print(f"========== Running {model_name} patch-to-patient Fold {fold_idx} ==========")
        train_val_patient_mil(
            model=model,
            loaders=loaders,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            cfg=cfg,
            params={"epoch": 0},
        )

    experiment_dir = os.path.join(cfg.model_dir, cfg.Task_name, f"seed-{cfg.seed}")
    threshold = write_pooled_validation_threshold(experiment_dir, folds=cfg.DATA.fold_K)
    print(f"Pooled validation threshold saved for {model_name}: {threshold:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Direct patch-to-patient MIL baseline runner")
    parser.add_argument("--models", default="ABMIL,TransMIL,MambaMIL", help="Comma-separated subset to run.")
    parser.add_argument("--abmil-cfg", default=BASELINE_CONFIGS["ABMIL"]["cfg"], type=str)
    parser.add_argument("--transmil-cfg", default=BASELINE_CONFIGS["TransMIL"]["cfg"], type=str)
    parser.add_argument("--mamba-cfg", default=BASELINE_CONFIGS["MambaMIL"]["cfg"], type=str)
    parser.add_argument("--task-prefix", default="MUSK_DirectMIL", type=str)
    parser.add_argument("--result-root", default=None, type=str)
    parser.add_argument("--seed", default=421, type=int)
    parser.add_argument("--gpu-id", default=None, type=int)
    parser.add_argument("--fold-start", default=None, type=int)
    parser.add_argument("--fold-end", default=None, type=int)
    parser.add_argument("--fold-k", default=None, type=int)
    parser.add_argument("--num-epochs", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--max-patches-per-patient", default=0, type=int)
    parser.add_argument("--cuda-visible-devices", default=None, type=str)
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cfg_paths = {
        "ABMIL": args.abmil_cfg,
        "TransMIL": args.transmil_cfg,
        "MambaMIL": args.mamba_cfg,
    }
    selected_models = _parse_csv_list(args.models)
    for model_name in selected_models:
        if model_name not in BASELINE_CONFIGS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {list(BASELINE_CONFIGS)}")
        _run_one_baseline(
            model_name=model_name,
            cfg_path=cfg_paths[model_name],
            model_type=BASELINE_CONFIGS[model_name]["model_type"],
            args=args,
        )


if __name__ == "__main__":
    main()
