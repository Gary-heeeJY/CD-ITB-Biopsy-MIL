import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .LADE_loss import LADELoss

#todo
def get_loss(cfg, cls_num_list, per_cls_weights):
    losses = {
        # "BCEloss" : BCEloss(cfg),
        "focalloss": MultiCEFocalLoss(cfg),
        # "MIL_BCEloss": MIL_BCEloss(cfg),
        # "MIL3_BCEloss": MIL3_BCEloss(cfg),
        "LDAM": LDAMLoss(cls_num_list, max_m=cfg.SOLVER.LDAM.max_m, s=cfg.SOLVER.LDAM.s, weight=per_cls_weights),
        "LADE": LADELoss(num_classes=cfg.DATA.n_classes, cls_num_list=cls_num_list, per_cls_weights=per_cls_weights,
                         device=cfg.gpu_id)
    }

    return losses.get(cfg.SOLVER.METRIC, None)


def get_case_cls_num(case_info_all):
    train_case_cls_num = []
    pN0_count, pN0_i_count, pN1mi_count, pN1_count, pN2_count = 0, 0, 0, 0, 0
    for case_name in list(case_info_all.keys()):
        case_info = case_info_all[case_name]
        case_label = case_info['case_label']
        if case_label.item() == 0:
            pN0_count += 1
        elif case_label.item() == 1:
            pN0_i_count += 1
        elif case_label.item() == 2:
            pN1mi_count += 1
        elif case_label.item() == 3:
            pN1_count += 1
        else:
            pN2_count += 1

    train_case_cls_num.append(pN0_count)
    train_case_cls_num.append(pN0_i_count)
    train_case_cls_num.append(pN1mi_count)
    train_case_cls_num.append(pN1_count)
    train_case_cls_num.append(pN2_count)
    return train_case_cls_num

class get_came_case_loss(nn.Module):
    def __init__(self, cfg, cls_num_list):
        super().__init__()
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(cfg.gpu_id)
        self.loss_func = LADELoss(num_classes=cfg.DATA.case_classes, cls_num_list=cls_num_list, per_cls_weights=per_cls_weights,
                         device=cfg.gpu_id)

    def forward(self, pred, label):
        loss_dict = {}
        loss = self.loss_func(pred, label)
        loss_dict['case_loss'] = loss['loss']

        return loss_dict


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  #nj的四次开方
        m_list = m_list * (max_m / np.max(m_list))     #常系数C
        m_list = torch.FloatTensor(m_list)        #转成tensor
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight    #和频率相关的re-weight
        self.n_classes = len(cls_num_list)

    def forward(self, x, target):
        device = x.device
        self.m_list = self.m_list.to(device)
        self.weight = self.weight.to(device)
        index = torch.zeros_like(x, dtype=torch.uint8)   #和x维度一致全为0的tensor

        index.scatter_(1, target.data.view(-1, 1), 1)
        # inst_labels = F.one_hot(target, num_classes=self.n_classes).squeeze()
        index_float = index.type(torch.cuda.FloatTensor)
        # index_float = torch.FloatTensor(index).to(device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m     #y的logits减去margin

        output = torch.where(index, x_m, x)    #合并矩阵 (x_m， x)， 根据index
        loss = F.cross_entropy(self.s*output, target, weight=self.weight)
        loss_dict = {}
        loss_dict['loss'] = loss
        return loss_dict


class get_case_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        loss_dict = {}
        loss = self.loss_func(pred, label)
        loss_dict['case_loss'] = loss

        return loss_dict


class CategoryCrossEntropy(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, pred, soft_label):
        soft_label = F.softmax(soft_label / self.T, -1)
        soft_label = Variable(soft_label.detach().data, requires_grad=False)
        loss = -1.0 * torch.sum(Variable(soft_label) * torch.log_softmax(pred / self.T, -1), dim=-1)
        loss = loss.mean(-1).mean(-1)
        return loss


class AttLoss(nn.Module):
    def __init__(self, s_factor):
        super().__init__()
        self.s = s_factor

    def forward(self, att):
        t = att.size(1)
        max_att_values, _ = torch.topk(att, max(int(t // self.s), 1), -1)
        mean_max_att = max_att_values.mean(1)

        min_att_values, _ = torch.topk(-att, max(int(t // self.s), 1), -1)
        mean_min_att = -min_att_values.mean(1)

        loss = (mean_min_att - mean_max_att).mean(0)

        return loss

class BCEloss(nn.Module):
    def __init__(self, cfg):
        super(BCEloss, self).__init__()
        self.BCEloss = torch.nn.BCELoss()

    def forward(self, batch_pre, batch_y):
        loss_dict = {}
        batch_pre = torch.sigmoid(batch_pre)
        BCEloss = self.BCEloss(batch_pre, batch_y)

        loss_dict['BCEloss'] = BCEloss
        loss_dict['loss'] = BCEloss * 1.0

        return loss_dict


class MultiCEFocalLoss(torch.nn.Module):
    # def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
    def __init__(self, cfg):
        super(MultiCEFocalLoss, self).__init__()
        self.class_num = cfg.DATA.n_classes
        if cfg.SOLVER.FocalLoss.alpha is None:
            self.alpha = Variable(torch.ones(self.class_num, 1))
        else:
            self.alpha = cfg.SOLVER.FocalLoss.alpha
        self.gamma = cfg.SOLVER.FocalLoss.gamma
        self.reduction = cfg.SOLVER.FocalLoss.reduction
        self.device = cfg.gpu_id

    def forward(self, predict, target):
        predict = predict.to(self.device)
        pt = F.softmax(predict, dim=1)  # softmax获取预测概率
        class_mask = F.one_hot(target, self.class_num).to(self.device)  # 获取target的one hot 编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].view(-1, 1).to(self.device)  # alpha 是给定的list (tensor), 里面的元素是每一类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask, 提取对应的pt
        log_p = probs.log()  # 原始ce上增加一个动态权重衰减因子   ce指的是交叉熵 crossentropy
        loss = - alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        loss_dict = {}
        if self.reduction == 'mean':
            focalloss = loss.mean()
            loss_dict['focalloss'] = focalloss
        elif self.reduction == 'sum':
            focalloss = loss.sum()
            loss_dict['focalloss'] = focalloss

        loss_dict['loss'] = focalloss * 1.0
        return loss_dict

#  Focalloss的目前不仅是为了控制样本不平衡的现象，还有个作用就是，让网络着重训练难样本
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt + 1e-9) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + 1e-9)

        loss_dict = {}
        if self.reduction == 'mean':
            focalloss = torch.mean(loss)
            loss_dict['focalloss'] = focalloss
        elif self.reduction == 'sum':
            focalloss = torch.sum(loss)
            loss_dict['focalloss'] = focalloss

        loss_dict['loss'] = focalloss*1.0
        return loss_dict

class MIL_BCEloss(nn.Module):
    def __init__(self, cfg):
        super(MIL_BCEloss, self).__init__()
        self.batch_loss = torch.nn.BCELoss()
        self.patch_loss = torch.nn.BCELoss()
        self.reg_loss = MSEDLoss()
        self.alpha = cfg.MODEL.MIL_CLUSTER.alpha
        self.beta = cfg.MODEL.MIL_CLUSTER.beta
        self.gama = cfg.MODEL.MIL_CLUSTER.gama

    def forward(self, batch_pre, batch_y, activate_func = 'sigmoid'):
        batch_pre, patch_pre, patch_att = batch_pre
        batch_y, patch_y, patch_cluster = batch_y

        loss_dict = {}
        if activate_func == 'sigmoid':
            batch_pre = torch.sigmoid(batch_pre)
            patch_pre = torch.sigmoid(patch_pre)

        loss_dict['batch_loss'] = self.batch_loss(batch_pre, batch_y)
        loss_dict['patch_loss'] = self.patch_loss(patch_pre, patch_y)
        loss_dict['reg_loss'] = self.reg_loss(patch_att, patch_cluster)

        loss_dict['loss'] = loss_dict['batch_loss'] * self.alpha + loss_dict['patch_loss'] * self.beta + loss_dict['reg_loss'] * self.gama

        return loss_dict



class MIL3_BCEloss(nn.Module):
    def __init__(self, cfg):
        super(MIL3_BCEloss, self).__init__()
        self.batch_loss = torch.nn.BCELoss()
        self.patch_loss = torch.nn.BCELoss()
        self.reg_loss = MSEDLoss()
        self.alpha = cfg.MODEL.MIL_CLUSTER.alpha
        self.beta = cfg.MODEL.MIL_CLUSTER.beta
        self.gama = cfg.MODEL.MIL_CLUSTER.gama
        self.activate_func = cfg.SOLVER.activate_func

    def forward(self, batch_pre, batch_y):
        batch_pre, patch_pre, patch_att = batch_pre
        batch_y, patch_y, patch_cluster = batch_y

        loss_dict = {}
        if self.activate_func == 'sigmoid':
            batch_pre = torch.sigmoid(batch_pre)
            patch_pre = torch.sigmoid(patch_pre)

        loss_dict['batch_loss'] = self.batch_loss(batch_pre, batch_y)
        loss_dict['patch_loss'] = self.patch_loss(patch_pre, patch_y)
        loss_dict['reg_loss'] = self.reg_loss(patch_att, patch_cluster)

        loss_dict['loss'] = loss_dict['batch_loss'] * self.alpha + loss_dict['patch_loss'] * self.beta + loss_dict[
            'reg_loss'] * self.gama

        return loss_dict

    def forward_patch(self, patch_pre, patch_y):
        loss_dict = {}
        if self.activate_func == 'sigmoid':
            patch_pre = torch.sigmoid(patch_pre)

        loss_dict['patch_loss'] = self.patch_loss(patch_pre, patch_y)
        loss_dict['loss'] = loss_dict['patch_loss'] * self.beta

        return loss_dict

    def forward_batch(self, batch_pre, batch_y):
        loss_dict = {}
        if self.activate_func == 'sigmoid':
            batch_pre = torch.sigmoid(batch_pre)

        loss_dict['batch_loss'] = self.batch_loss(batch_pre, batch_y)
        loss_dict['loss'] = loss_dict['batch_loss'] * self.alpha

        return loss_dict


class KLDLoss(nn.Module): #TODO
    """KL-divergence loss between attention weight and uniform distribution"""

    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, attn_val, cluster):
        """
        Example:
          Input - attention value = torch.tensor([0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05,
                                0.1, 0.05, 0.1, 0.05, 0.1, 0.05])
                  cluster = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
          Output - 0.0043
        """
        kld_loss = 0
        is_cuda = attn_val.device
        if isinstance(cluster, torch.Tensor) :
            cluster = cluster.cpu().numpy()
            cluster = cluster.flatten()

        for cls in np.unique(cluster):
            index = np.where(cluster == cls)[0]
            # HARD CODE - if number of images less than 4 in a cluster then skip
            if len(index) <= 4:
                continue
            kld_loss += F.kl_div(F.log_softmax(attn_val[index], dim=0)[None], \
                                 torch.ones(len(index), 1)[None].to(is_cuda) / len(index), \
                                 reduction='batchmean')

        return kld_loss


class MSEDLoss(nn.Module): #TODO
    """MSE loss between attention weight and uniform distribution"""

    def __init__(self):
        super(MSEDLoss, self).__init__()
        self.MSELoss = torch.nn.MSELoss()

    def forward(self, attn_val, cluster):
        """
        Example:
          Input - attention value = torch.tensor([0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05,
                                0.1, 0.05, 0.1, 0.05, 0.1, 0.05])
                  cluster = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
          Output - 0.x
        """
        loss = 0
        is_cuda = attn_val.device
        if isinstance(cluster, torch.Tensor) :
            cluster = cluster.cpu().numpy()
            cluster = cluster.flatten()

        for cls in np.unique(cluster):
            index = np.where(cluster == cls)[0]
            # HARD CODE - if number of images less than 4 in a cluster then skip
            if len(index) <= 4:
                continue
            loss += self.MSELoss(attn_val[index][None], \
                                 torch.ones(len(index), 1)[None].to(is_cuda) / len(index))

        return loss