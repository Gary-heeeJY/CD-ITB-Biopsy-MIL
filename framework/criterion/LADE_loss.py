import functools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils_ import *


class LADELoss(nn.Module):
    def __init__(self, num_classes=10, cls_num_list=None, per_cls_weights=None, remine_lambda=0.1, device=None):
        super().__init__()

        self.img_num_per_cls = torch.tensor(np.array(cls_num_list)).float().to(device)
        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()

        self.balanced_prior = torch.tensor(1. / num_classes).float().to(device)
        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        # self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float())).to(device)
        self.cls_weight = per_cls_weights

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        loss_dict = {}
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        loss_dict['loss'] = loss
        return loss_dict


def create_loss(num_classes, img_max=None, prior=None, prior_txt=None, remine_lambda=0.1):
    print("Loading LADELoss.")
    return LADELoss(
        num_classes=num_classes,
        img_max=img_max,
        prior=prior,
        prior_txt=prior_txt,
        remine_lambda=remine_lambda,
    )