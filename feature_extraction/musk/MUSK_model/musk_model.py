# musk_model.py
import torch.nn as nn
import logging
from timm.models import create_model
from config import config
import torch
import torch.nn as nn
import pandas as pd
import os
import openslide
from torch.utils.data import Dataset
from timm.models import create_model
import utils
from transformers import XLMRobertaTokenizer
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import chardet
import logging
from timm.models import create_model
import h5py
import numpy as np

logger = logging.getLogger(__name__)

class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info("创建MUSK模型...")
        self.musk = create_model(config.MODEL_NAME, pretrained=False)
        utils.load_model_and_may_interpolate(config.MUSK_MODEL_LOCAL_PATH, self.musk, 'model|module', '')

        # Feature extraction is frozen and runs in float32 for reproducibility.
        logger.info("使用单精度浮点数 (float32)")
        self.musk = self.musk.to(config.DEVICE).float().eval()  # 移除.half()

        # 从模型配置中获取特征维度
        embed_dim = self.musk.args.encoder_embed_dim
        self.vision_dim = embed_dim ## * len(config.SCALES)
        self.text_dim = embed_dim

        # Keep the legacy classifier definition for compatibility with the upstream model.
        self.classifier = nn.Sequential(
            nn.Linear(self.vision_dim + self.text_dim, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 2)
        ).to(config.DEVICE).float()

    def forward(self, images, txt_ids, padding_mask):
        # 添加文本输入类型转换
        images = images.float()

        # 冻结MUSK权重
        with torch.no_grad():
            # 图像特征
            img_features, _ = self.musk(
                image=images,
                return_global=True,
                with_head=False,
                out_norm=False,
                ms_aug=False
            )

            # 文本特征
            _, txt_features = self.musk(
                text_description=txt_ids,
                padding_mask=padding_mask,
                return_global=True,
                with_head=False,
                out_norm=False
            )

        # 特征融合 (转换为float32)
        img_features = img_features.float()
        txt_features = txt_features.float()
        fusion = torch.cat([img_features, txt_features], dim=1)

        # 分类预测
        return self.classifier(fusion)
