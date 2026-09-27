import torch
from .GradualWarmupScheduler import GradualWarmupScheduler
from torch.optim import lr_scheduler

def build_optimizer(cfg, params):
    # Adam
    # optimizer = torch.optim.Adam(params['params'], params['lr'], [cfg.SOLVER.beta1, cfg.SOLVER.beta2],
    #                              weight_decay=cfg.SOLVER.weight_decay)
    # optimizer2 = torch.optim.Adam(params['case_params'], params['case_lr'], [cfg.SOLVER.beta1, cfg.SOLVER.beta2],
    #                               weight_decay=cfg.SOLVER.weight_decay)

    # AdamW
    optimizer = torch.optim.AdamW(params['params'], params['lr'], [cfg.SOLVER.beta1, cfg.SOLVER.beta2],
                                  weight_decay=cfg.SOLVER.weight_decay)
    optimizer2 = torch.optim.AdamW(params['case_params'], params['case_lr'], [cfg.SOLVER.beta1, cfg.SOLVER.beta2],
                                   weight_decay=cfg.SOLVER.weight_decay)

    # SGD
    # optimizer = torch.optim.SGD(params['params'], params['lr'], momentum=cfg.SOLVER.momentum,
    #                             weight_decay=cfg.SOLVER.weight_decay)
    # optimizer2 = torch.optim.SGD(params['case_params'], params['case_lr'], momentum=cfg.SOLVER.momentum,
    #                              weight_decay=cfg.SOLVER.weight_decay)
    return optimizer, optimizer2


def build_lr_scheduler(cfg, optimizer):
    # lr schachle策略(要传入optimizer才可以)
    # 暂时的三种情况,(1)只用cos,(2)只用warmup,(3)两者都用
    if cfg.SOLVER.lr_warm_epoch != 0 and cfg.SOLVER.lr_cos_epoch == 0:
        cfg.SOLVER.update_lr(cfg.SOLVER.lr_low)  # 使用warmup需要吧lr初始化为最小lr
        lr_sch = GradualWarmupScheduler(optimizer,
                                             multiplier=cfg.SOLVER.lr / cfg.SOLVER.lr_low,
                                             total_epoch=cfg.SOLVER.lr_warm_epoch,
                                             after_scheduler=None)
        print('use warmup lr sch')
    elif cfg.SOLVER.lr_warm_epoch == 0 and cfg.SOLVER.lr_cos_epoch != 0:
        lr_sch = lr_scheduler.CosineAnnealingLR(optimizer,
                                                     cfg.SOLVER.lr_cos_epoch,
                                                     eta_min=cfg.SOLVER.lr_low)
        print('use cos lr sch')
    elif cfg.SOLVER.lr_warm_epoch != 0 and cfg.SOLVER.lr_cos_epoch != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.SOLVER.lr_low
        scheduler_cos = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.lr_cos_epoch,
            eta_min=cfg.SOLVER.lr_low,
        )
        lr_sch = GradualWarmupScheduler(optimizer,
                                             multiplier=cfg.SOLVER.lr / cfg.SOLVER.lr_low,
                                             total_epoch=cfg.SOLVER.lr_warm_epoch,
                                             after_scheduler=scheduler_cos)
        print('use warmup and cos lr sch')
    else:
        lr_sch = lr_scheduler.StepLR(optimizer,
                                     step_size=cfg.SOLVER.num_epochs_decay,
                                     gamma=cfg.SOLVER.decay_ratio)
        print('use step lr sch')

    return lr_sch, optimizer
p_size=cfg.SOLVER.num_epochs_decay,
                                     gamma=cfg.SOLVER.decay_ratio)
        print('use step lr sch')

    return lr_sch, optimizer