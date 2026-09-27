import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.wsi_dataset import Wsi_Region
from utils.transform_utils import get_eval_transforms
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore
from utils.constants import MODEL2CONSTANTS
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile


def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level=-1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)

    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap


def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object


def compute_from_patches(wsi_object, img_transforms, feature_extractor=None, clam_pred=None, model=None, batch_size=512,
                         attn_save_path=None, ref_scores=None, feat_save_path=None, override_coords=None,
                         **wsi_kwargs):
    top_left = wsi_kwargs.get('top_left', None)
    bot_right = wsi_kwargs.get('bot_right', None)

    # 统一 patch_size 格式
    patch_size = wsi_kwargs.get('patch_size', 512)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    # 坐标复用逻辑优化
    if override_coords is not None:
        print(f"Using {len(override_coords)} overridden coordinates from training data.")

        class SimplePatchDataset(torch.utils.data.Dataset):
            def __init__(self, wsi, coords, transforms, level, size):
                self.wsi = wsi
                self.coords = coords
                self.transforms = transforms
                self.level = level
                self.size = size  # (256, 256)

            def __len__(self):
                return len(self.coords)

            def __getitem__(self, idx):
                coord = self.coords[idx]
                # 读取图片
                img = self.wsi.read_region(tuple(coord), self.level, self.size).convert('RGB')
                if self.transforms:
                    img = self.transforms(img)
                # 【关键修改】必须 unsqueeze(0) 以适配 CLAM 的 collate_features
                return img.unsqueeze(0), coord

        roi_dataset = SimplePatchDataset(wsi_object.wsi, override_coords, img_transforms,
                                         wsi_kwargs.get('level', 0), patch_size)
    else:
        # 原有逻辑
        roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)

    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)

    # 预处理 ref_scores
    if ref_scores is not None:
        ref_scores = np.array(ref_scores).flatten()

    mode = "w"
    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
        roi = roi.to(device)
        coords = coords.numpy()

        with torch.inference_mode():
            features = feature_extractor(roi)

            if attn_save_path is not None:
                A = model(features, attention_only=True)

                if A.size(0) > 1:
                    A = A[clam_pred]

                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        score_val = A[score_idx].item()
                        A[score_idx] = score2percentile(score_val, ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object
