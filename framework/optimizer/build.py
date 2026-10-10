import torch
from torch.optim import lr_scheduler

from .GradualWarmupScheduler import GradualWarmupScheduler


def build_optimizer(cfg, params):
    optimizer = torch.optim.AdamW(
        params["params"],
        params["lr"],
        [cfg.SOLVER.beta1, cfg.SOLVER.beta2],
        weight_decay=cfg.SOLVER.weight_decay,
    )
    optimizer2 = torch.optim.AdamW(
        params["case_params"],
        params["case_lr"],
        [cfg.SOLVER.beta1, cfg.SOLVER.beta2],
        weight_decay=cfg.SOLVER.weight_decay,
    )
    return optimizer, optimizer2


def build_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.lr_warm_epoch != 0 and cfg.SOLVER.lr_cos_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg.SOLVER.lr_low
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.SOLVER.lr / cfg.SOLVER.lr_low,
            total_epoch=cfg.SOLVER.lr_warm_epoch,
            after_scheduler=None,
        )
        print("use warmup lr scheduler")
    elif cfg.SOLVER.lr_warm_epoch == 0 and cfg.SOLVER.lr_cos_epoch != 0:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            cfg.SOLVER.lr_cos_epoch,
            eta_min=cfg.SOLVER.lr_low,
        )
        print("use cosine lr scheduler")
    elif cfg.SOLVER.lr_warm_epoch != 0 and cfg.SOLVER.lr_cos_epoch != 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg.SOLVER.lr_low
        cosine = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.lr_cos_epoch,
            eta_min=cfg.SOLVER.lr_low,
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.SOLVER.lr / cfg.SOLVER.lr_low,
            total_epoch=cfg.SOLVER.lr_warm_epoch,
            after_scheduler=cosine,
        )
        print("use warmup and cosine lr scheduler")
    else:
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.SOLVER.num_epochs_decay,
            gamma=cfg.SOLVER.decay_ratio,
        )
        print("use step lr scheduler")
    return scheduler, optimizer
