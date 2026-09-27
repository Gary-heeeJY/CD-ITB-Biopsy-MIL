import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils_.evaluation import youden_threshold


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


def _compute_auc(labels, probs, n_classes):
    try:
        probs = np.asarray(probs)
        if n_classes == 2:
            return roc_auc_score(labels, probs[:, 1])
        return roc_auc_score(labels, probs, multi_class="ovr")
    except Exception:
        return 0.5


def _run_phase(model, loader, criterion, optimizer, device, n_classes, train):
    model.train(train)
    total_loss = 0.0
    probs_list, preds_list, labels_list = [], [], []

    for feat, label, case_id, slide_ids in loader:
        feat = feat.to(device)
        label = label.to(device)

        with torch.set_grad_enabled(train):
            logits = _model_logits(model(feat))
            loss = criterion(logits, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        probs_list.extend(probs)
        preds_list.extend(preds)
        labels_list.extend(label.detach().cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(labels_list, preds_list)
    f1 = f1_score(labels_list, preds_list, average="macro", zero_division=0)
    auc = _compute_auc(labels_list, probs_list, n_classes)
    threshold = (
        youden_threshold(labels_list, np.asarray(probs_list)[:, 1])
        if n_classes == 2 and not train
        else 0.5
    )
    positive_scores = np.asarray(probs_list)[:, 1] if probs_list and n_classes == 2 else np.asarray([])
    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "labels": np.asarray(labels_list, dtype=np.int64),
        "positive_scores": positive_scores.astype(np.float64),
    }


def train_val_patient_mil(model, loaders, criterion, lr_scheduler, optimizer, cfg, params=None):
    logger = logging.getLogger("train")
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_epoch = cfg.SOLVER.num_epochs
    n_classes = cfg.DATA.n_classes
    start_epoch = 0 if params is None else params.get("epoch", 0)
    best_auc = float("-inf")
    best_score = float("-inf")

    for epoch in range(start_epoch, n_epoch):
        phases = ["train"]
        if "eval" in loaders and epoch % cfg.SOLVER.val_step == 0:
            phases.append("eval")

        for phase in phases:
            metrics = _run_phase(
                model=model,
                loader=loaders[phase],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                n_classes=n_classes,
                train=(phase == "train"),
            )

            logger.info(
                "[%s] Epoch:%d loss:%.4f auc:%.4f acc:%.4f f1:%.4f",
                phase,
                epoch,
                metrics["loss"],
                metrics["auc"],
                metrics["acc"],
                metrics["f1"],
            )
            print(
                f"[{phase}] Epoch:{epoch} | "
                f"Loss:{metrics['loss']:.4f} AUC:{metrics['auc']:.4f} "
                f"ACC:{metrics['acc']:.4f} F1:{metrics['f1']:.4f}"
            )

            if phase == "eval":
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                }
                torch.save(state, os.path.join(cfg.model_dir_, "last_model.pt"))

                if metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    torch.save(state, os.path.join(cfg.model_dir_, "patient_auc_best_model.pt"))
                    np.savez_compressed(
                        os.path.join(cfg.model_dir_, "validation_predictions.npz"),
                        labels=metrics["labels"],
                        positive_scores=metrics["positive_scores"],
                    )

                joint_score = 0.5 * metrics["auc"] + 0.25 * metrics["acc"] + 0.25 * metrics["f1"]
                if joint_score > best_score:
                    best_score = joint_score
                    torch.save(state, os.path.join(cfg.model_dir_, "patient_best_model.pt"))

        lr_scheduler.step()

    if "test" in loaders:
        checkpoint_path = os.path.join(cfg.model_dir_, "patient_auc_best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
        metrics = _run_phase(
            model=model,
            loader=loaders["test"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_classes=n_classes,
            train=False,
        )
        logger.info(
            "[test] loss:%.4f auc:%.4f acc:%.4f f1:%.4f",
            metrics["loss"],
            metrics["auc"],
            metrics["acc"],
            metrics["f1"],
        )
        print(
            f"[test] Loss:{metrics['loss']:.4f} AUC:{metrics['auc']:.4f} "
            f"ACC:{metrics['acc']:.4f} F1:{metrics['f1']:.4f}"
        )

    logger.info("Patient-level MIL training finished.")
