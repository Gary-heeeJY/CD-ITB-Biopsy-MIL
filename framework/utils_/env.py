import os
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#todo just for single gpu
def init_env(cfg):
    if not cfg.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)