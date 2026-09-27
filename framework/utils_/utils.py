import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
# from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
from torch.autograd import Variable
import math
from itertools import islice
import collections
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if device.type == 'cuda':
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    id = [item[2] for item in batch]
    return [img, label, id]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
    kwargs = {'num_workers': 0, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SequentialSampler(dataset),
                        collate_fn=collate_features, **kwargs)
    return loader


def get_split_loader(split_dataset, training=False, testing=False, weighted=False):
    """
        return either the validation loader or training loader
    """
    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)),
                                    collate_fn=collate_MIL, **kwargs)
            # loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset), collate_fn=collate_MIL, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset),
                                    collate_fn=collate_MIL, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset),
                                collate_fn=collate_MIL, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate_MIL,
                            **kwargs)

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            # for c in range(len(test_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids
            # test_ids = np.random.choice(possible_indices, test_num[c], replace=False)
            remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
            # remaining_ids = np.setdiff1d(possible_indices, test_ids)
            all_val_ids.extend(val_ids)
            # all_test_ids.extend(test_ids)

            if custom_test_ids is None:  # sample test split

                val_ids = np.random.choice(remaining_ids, val_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, val_ids)
                all_val_ids.extend(val_ids)
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()
            # init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        self.class_num = class_num
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predict, target):
        device = predict.device
        pt = F.softmax(predict, dim=1)  # softmax获取预测概率
        class_mask = F.one_hot(target, self.class_num).to(device)  # 获取target的one hot 编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].view(-1, 1).to(device)  # alpha 是给定的list (tensor), 里面的元素是每一类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask, 提取对应的pt
        log_p = probs.log()  # 原始ce上增加一个动态权重衰减因子   ce指的是交叉熵 crossentropy
        loss = - alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def adjust_learning_lr(optimizer, epoch, lr):
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 80:
        lr = lr * 0.01
    elif epoch > 160:
        lr = lr * 0.001
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def rampup(global_step, rampup_length=40):
    if global_step < rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)


def rampdown(epoch, num_epochs=200, rampdown_length=50):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return np.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0


def step_rampup(epoch, rampup_length=80):
    if epoch <= rampup_length:
        return 1.0
    else:
        return 0.0