import os
import logging
import torch
import sys
from fvcore.common.file_io import PathManager
from collections import deque, defaultdict
import numpy as np
from scipy import stats

def setup_logger(name="test_ensemble", output=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) # 设置为 INFO 或 DEBUG
    logger.propagate = False

    # 清除已有的 handlers，防止重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    plain_formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s %(message)s', datefmt="%m/%d %H:%M:%S")

    # 1. 添加文件 Handler (使用 'w' 模式覆盖旧日志，避免混淆)
    if output is not None:
        try:
            os.makedirs(os.path.dirname(output), exist_ok=True)
            # 使用 'w' 模式，每次运行清空旧日志
            fh = logging.FileHandler(output, mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Failed to setup file logger: {e}")

    # 2. 添加控制台 Handler (让你在屏幕上也能直接看到结果，不用去翻文件)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    return logger


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=1):
        self.fmt = "{median:.4f}({average:.4f})"
        # self.deque = deque(maxlen=window_size)
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.ci_alpha = 0.95

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def reset(self):
        self.deque = []
        self.total = 0.0
        self.count = 0

    @property
    def GetMedianValue(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def GetAverageValue(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def GetSTDValue(self):
        d = torch.tensor(list(self.deque))
        return d.std().item()

    @property
    def GetConfidenceIntervalValue(self):
        ci = stats.t.interval(self.ci_alpha, self.count-1, loc=self.GetAverageValue, scale=self.GetSTDValue/np.sqrt(self.count))
        return ci

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return  self.fmt.format(
            median = self.GetMedianValue,
            average = self.GetAverageValue
        )

class Losslogger1(object):
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def updata(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def reset(self, key):
        self.meters[key].reset()

    def GetValue(self, key=None, attr=None):
        return self.meters[key].__getattribute__(attr)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

class Losslogger(object):
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def updata(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def reset(self, key):
        self.meters[key].reset()

    def GetValue(self, key=None, attr=None):
        return self.meters[key].__getattribute__(attr)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=40, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss