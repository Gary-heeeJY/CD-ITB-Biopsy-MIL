from criterion.metrics import get_metric, get_patient_feat, get_came_patient_metric, get_biospy_patient_metric, \
    patient_feat_feedback, get_slide_metric, get_biospy_patient_metric_ori, get_came_patient_metric_ori
from criterion.losses import get_loss, get_case_loss
from criterion.losses import AttLoss, CategoryCrossEntropy
import numpy as np
import torch
import torch.nn as nn


def build_criterion(cfg, cls_num_list):
    beta = 0.9999
    # effective_num = 1.0 - np.power(beta, cls_num_list)
    # per_cls_weights = (1.0 - beta) / np.array(effective_num)
    # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    # per_cls_weights = torch.FloatTensor(per_cls_weights).to(cfg.gpu_id)

    effective_num = 1.0 - np.power(beta, cls_num_list)
    effective_num = np.where(effective_num == 0, 1e-10, effective_num)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(cfg.gpu_id)

    # per_cls_weights = torch.ones(len(cls_num_list)).to(cfg.gpu_id) # 设置为1即不使用类别权重

    print('per_cls_weights:', per_cls_weights, len(cls_num_list))
    criterion = {}
    # criterion['attn_loss'] = AttLoss(cfg.MODEL.KSRP.s_factor)
    criterion['pseudo_label_loss'] = CategoryCrossEntropy(cfg.MODEL.KSRP.T)
    criterion['slide_loss'] = get_loss(cfg, cls_num_list, per_cls_weights)
    criterion['case_loss'] = get_case_loss()
    criterion['metric1'] = get_metric(cfg)
    criterion['metric2'] = get_slide_metric(cfg)
    criterion['case_info'] = get_patient_feat(cfg)
    criterion['case_feat_feedback'] = patient_feat_feedback(cfg)
    if cfg.DATA.type == 'biospy_CD_ITB':
        criterion['case_metric'] = get_biospy_patient_metric(cfg)
        criterion['case_metric_ori'] = get_biospy_patient_metric_ori(cfg)
    else:
        criterion['case_metric'] = get_came_patient_metric(cfg)
        criterion['case_metric_ori'] = get_came_patient_metric_ori(cfg)

    return criterion

def build_criterion_WCLS(cfg, cls_num_list):
    """
    针对 CD/TB (1.7:1) 和 Slide (3.6:1) 的轻微不平衡，
    使用 Weighted CrossEntropy + Label Smoothing 是最佳方案。
    """

    # 1. 计算类别权重 (用于解决 Slide 级 3.6:1 的不平衡)
    # 使用简单的逆频率权重 (Inverse Frequency) 即可，不需要复杂的 beta 衰减
    # 你的 slide 数据: CD 327, TB 91, NO 221
    # 权重逻辑: 数量越少，权重越大

    # 转换为 float 避免整数除法
    counts = np.array(cls_num_list, dtype=np.float32)

    # 加上微小值防止除零 (虽然你的数据不会为0)
    weights = 1.0 / (counts + 1e-6)

    # 归一化权重，让它们均值为 1 (保持 Loss 数值尺度稳定)
    weights = weights / np.mean(weights)

    # 转为 Tensor
    per_cls_weights = torch.FloatTensor(weights).to(cfg.gpu_id)

    print(f'Slide Counts: {cls_num_list}')
    print(f'Calculated Weights: {per_cls_weights}')
    # The inverse-frequency weights follow the class counts supplied by the loader.

    criterion = {}

    # ============================================================
    # 【核心修改】 Weighted CE + Label Smoothing
    # ============================================================
    criterion['slide_loss'] = nn.CrossEntropyLoss(
        weight=per_cls_weights,  # 解决不平衡
        label_smoothing=0.1  # 解决泛化差/过拟合
    )
    # ============================================================

    criterion['case_loss'] = get_case_loss()

    # ... 保留原本的 metric 初始化 ...
    criterion['metric1'] = get_metric(cfg)
    criterion['metric2'] = get_slide_metric(cfg)
    criterion['case_info'] = get_patient_feat(cfg)
    criterion['case_feat_feedback'] = patient_feat_feedback(cfg)

    if cfg.DATA.type == 'biospy_CD_ITB':
        criterion['case_metric'] = get_biospy_patient_metric(cfg)
        criterion['case_metric_ori'] = get_biospy_patient_metric_ori(cfg)
    else:
        criterion['case_metric'] = get_came_patient_metric(cfg)
        criterion['case_metric_ori'] = get_came_patient_metric_ori(cfg)

    return criterion
