from __future__ import print_function
import argparse
import sys
import os
import glob
import copy

# ================= [路径设置] =================
# 本脚本与其依赖包 (models / vis_utils / wsi_core / utils / dataset_modules)
# 都位于 Heatmap 目录下，因此只需把脚本所在目录加入搜索路径即可。
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# =============================================

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from utils.utils import *
import h5py
import yaml  # 仅在使用 --config_file 时用到，可不用
from vis_utils.heatmap_utils import drawHeatmap
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from tqdm import tqdm
from models.MambaMIL import MambaMIL
from wsi_core.WholeSlideImage import WholeSlideImage
import torch.nn.functional as F
from PIL import Image, ImageDraw
from dataset_modules.dataset_generic import generic_locate_file


# =========================================================================
# [配置区] 原来写在 heatmap_config.yaml 里的所有内容，现在直接写在这里。
# 想改路径 / 名单 / 参数，直接改下面 CONFIG，或用命令行参数覆盖。
# 服务器上是 Linux 路径(以 / 开头)；若在本机 Windows 测试，记得改成 Z:/...
# =========================================================================
CONFIG = {
    # ---- 实验与输出 ----
    'exp_arguments': {
        'n_classes': 3,                                   # CD / ITB / NO 三分类
        'save_exp_code': 'MambaMIL_Heatmap',              # 结果子目录名
        # 三类结果输出根目录（可用 --output_dir 一键覆盖）
        'production_save_dir': './outputs/MUSK_MambaMIL_DynamicWeight/seed-421/Heatmaps/production',
        'raw_save_dir':       './outputs/MUSK_MambaMIL_DynamicWeight/seed-421/Heatmaps/raw',
        'result_dir':         './outputs/MUSK_MambaMIL_DynamicWeight/seed-421/Heatmaps',
    },

    # ---- 数据来源 ----
    'data_arguments': {
        'data_dir': './data/wsi',  # 原始 WSI 目录
        'data_dir_key': 'source',
        'slide_ext': '.tif',
        'label_dict': {'CD': 1, 'ITB': 0, 'TB': 0, 'NO': 2},
        'process_list': None,   # 若填 CSV 路径，则沿用旧逻辑读取 slide_id 列；一般留 None，用下面的 patient_list
        'existing_patch_dir': './data/patches',  # patch 坐标 .h5
        'pre_extracted_feature_dir': './data/features/MUSK',  # 预提取特征 .pt
    },

    # ---- 模型 / 权重 ----
    'model_arguments': {
        # 存放 fold-0 ~ fold-(n_folds-1) 的目录（可用 --ckpt_dir 覆盖）
        'ckpt_dir': './outputs/MUSK_MambaMIL_DynamicWeight/seed-421',
        'model_name': 'case_best_model.pt',  # 每个 fold 目录下的权重文件名（可用 --model_name 覆盖）
        'model_type': 'mambamil',
        # 下面这些超参必须和你训练时 MambaMIL 的结构一致（可用 --in_dim 等覆盖）
        'in_dim': 1024,            # 特征维度（MUSK 为 1024）
        'n_classes': 3,
        'mamba_type': 'SRMamba',   # 'SRMamba' / 'Mamba' / 'BiMamba'
        'layer': 2,
        'act': 'gelu',
        'dropout': 0.0,
    },

    # ---- Patch 切分 ----
    'patching_arguments': {
        'patch_size': 512,
        'overlap': 0.0,
        'patch_level': 0,
        'custom_downsample': 1,
    },

    # ---- 可视化 / 热图 ----
    'heatmap_arguments': {
        'map_type': 'attention',
        'vis_level': 1,          # 多层级切片的可视化层级（10x 视图）；单层切片会走手动绘制分支
        'alpha': 0.5,            # 热图透明度
        'blank_canvas': False,
        'binarize': False,
        'binary_thresh': -1,
        'custom_downsample': 4,  # 热图额外降采样
        'cmap': 'jet',           # 颜色映射
        'save_ext': 'jpg',
        'use_ref_scores': True,
        'blur': True,            # 平滑
        'use_center_shift': True,
        'use_roi': False,
        'calc_heatmap': True,
        'save_orig': True,       # 同时保存原图缩略
    },

    # ---- 采样代表性 patch ----
    'sample_arguments': {
        'samples': [
            {'name': 'topk_high_attention', 'sample': True, 'k': 10, 'mode': 'topk', 'seed': 1},
        ],
    },

    # ---- 自定义患者名单（优先级高于 data_arguments.process_list）----
    # 只支持 .csv 文件：自动读取其中的 'slide_id' 列（若该列不存在则取第一列）。
    # 例如：'slide_id' 这一列写好要画热图的切片名（不含扩展名）。
    # 若此项为 None，则回退到 data_arguments.process_list（同样是一个 CSV）。
    # 命令行用 --patient_list 可临时覆盖此项。
    'patient_list': "./data/labels/internal_train.csv",
}


# ================= 患者名单加载 =================
def load_patient_list(patient_list):
    """patient_list 必须是一个 .csv 文件路径，自动读取其中的 'slide_id' 列
    （若没有该列则取第一列）。返回 slide_id 字符串列表。

    若 patient_list 为 None，返回 None，由主流程回退到 data_arguments.process_list。
    """
    if patient_list is None:
        return None
    if not os.path.isfile(patient_list):
        raise FileNotFoundError(
            "patient_list 必须是一个 .csv 文件路径，但未找到该文件：{}".format(patient_list)
        )
    df = pd.read_csv(patient_list)
    col = 'slide_id' if 'slide_id' in df.columns else df.columns[0]
    return [str(x) for x in df[col].dropna().tolist()]


# ================= 配置合并 =================
def build_config(args):
    """以内置 CONFIG 为基础，依次用可选 YAML 与命令行参数覆盖。"""
    cfg = copy.deepcopy(CONFIG)

    # 可选：用 YAML 覆盖（仅作为补充，不是必须）
    if getattr(args, 'config_file', None) and os.path.exists(args.config_file):
        ycfg = yaml.safe_load(open(args.config_file, 'r'))
        for k, v in ycfg.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        print(f"Loaded override config: {args.config_file}")

    # 命令行覆盖（只在显式提供时生效）
    if getattr(args, 'ckpt_dir', None):
        cfg['model_arguments']['ckpt_dir'] = args.ckpt_dir
    if getattr(args, 'model_name', None):
        cfg['model_arguments']['model_name'] = args.model_name
    if getattr(args, 'in_dim', None):
        cfg['model_arguments']['in_dim'] = args.in_dim
    if getattr(args, 'exp_code', None):
        cfg['exp_arguments']['save_exp_code'] = args.exp_code
    if getattr(args, 'data_dir', None):
        cfg['data_arguments']['data_dir'] = args.data_dir
    if getattr(args, 'patch_dir', None):
        cfg['data_arguments']['existing_patch_dir'] = args.patch_dir
    if getattr(args, 'feature_dir', None):
        cfg['data_arguments']['pre_extracted_feature_dir'] = args.feature_dir
    if getattr(args, 'vis_level', None) is not None:
        cfg['heatmap_arguments']['vis_level'] = args.vis_level
    if getattr(args, 'cmap', None):
        cfg['heatmap_arguments']['cmap'] = args.cmap
    if getattr(args, 'alpha', None) is not None:
        cfg['heatmap_arguments']['alpha'] = args.alpha
    if getattr(args, 'overlap', None) is not None:
        cfg['patching_arguments']['overlap'] = args.overlap
    if getattr(args, 'patient_list', None):
        cfg['patient_list'] = args.patient_list

    # 统一输出根目录：output_dir 即 Heatmaps 根目录，production/raw 为其子目录，
    # CSV 结果直接落在根目录（与旧版 create_heatmaps.py 的输出布局保持一致）
    if getattr(args, 'output_dir', None):
        cfg['exp_arguments']['production_save_dir'] = os.path.join(args.output_dir, 'production')
        cfg['exp_arguments']['raw_save_dir'] = os.path.join(args.output_dir, 'raw')
        cfg['exp_arguments']['result_dir'] = args.output_dir

    return cfg


# ================= 包装类定义 =================
class MambaMIL_Heatmap_Wrapper(nn.Module):
    def __init__(self, mamba_model):
        super().__init__()
        self.model = mamba_model

    def forward(self, x, attention_only=False):
        logits, h_slide, A = self.model(x)
        if len(A.shape) == 3:
            A = A.squeeze(0).permute(1, 0)
        elif len(A.shape) == 2:
            A = A.transpose(0, 1)
        if attention_only:
            return A
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        return logits, Y_prob, Y_hat, A, None


# ================= 核心辅助函数 =================
def inject_dummy_contours(wsi_object):
    w, h = wsi_object.wsi.level_dimensions[0]
    pt1, pt2, pt3, pt4 = [0, 0], [w, 0], [w, h], [0, h]
    full_slide_contour = np.array([pt1, pt2, pt3, pt4], dtype=np.int32).reshape(-1, 1, 2)
    contours = [full_slide_contour]
    holes = [[]]
    wsi_object.contours_tissue = contours
    wsi_object.contours_holes = holes
    wsi_object.holes_tissue = holes
    wsi_object.tissue_holes = holes
    return wsi_object


def infer_ensemble_slide(models_list, features, label, reverse_label_dict, k=1):
    features = features.to(device)
    if len(features.shape) == 2:
        features = features.unsqueeze(0)

    A_list = []
    prob_list = []

    with torch.inference_mode():
        for model in models_list:
            logits, Y_prob, _, A, _ = model(features)
            A_list.append(A.cpu().numpy())
            prob_list.append(Y_prob.cpu().numpy())

    A_ensemble = np.mean(np.array(A_list), axis=0)
    prob_ensemble = np.mean(np.array(prob_list), axis=0)

    Y_hat = np.argmax(prob_ensemble)
    Y_prob_final = torch.from_numpy(prob_ensemble).to(device)

    print('Ensemble Pred: {}, Label: {}, Prob: {}'.format(
        reverse_label_dict[Y_hat], label,
        ["{:.4f}".format(p) for p in prob_ensemble.flatten()]
    ))

    probs, ids = torch.topk(Y_prob_final, k)
    probs = probs[-1].cpu().numpy()
    ids = ids[-1].cpu().numpy()
    preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A_ensemble


# ================= [新增] 手动绘制热图函数 =================
# 专门用于处理只有 Level 0 的巨型 Slide，避免 RGBA 条纹问题
def manual_draw_heatmap(wsi_object, coords, A, save_path, patch_size_l0, downsample_factor=32, cmap='jet', alpha=0.5):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # 1. 生成背景缩略图 (替代 read_region)
    w, h = wsi_object.wsi.level_dimensions[0]
    target_size = (w // downsample_factor, h // downsample_factor)

    print(f"  -> Generating thumbnail for background (Target: {target_size})...")
    # get_thumbnail 是 OpenSlide 的内置优化方法，不会产生内存条纹
    img = wsi_object.wsi.get_thumbnail(target_size).convert('RGB')

    # 2. 准备绘制层 (Overlay)
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 3. 准备颜色映射
    cmap_obj = cm.get_cmap(cmap)

    # 归一化 Attention Scores 到 [0, 1] 用于颜色映射
    # 注意：这里我们做 Percentile 归一化，让颜色对比更明显
    a_scores = A.flatten()
    min_val, max_val = np.percentile(a_scores, 1), np.percentile(a_scores, 99)
    norm_scores = (a_scores - min_val) / (max_val - min_val + 1e-8)
    norm_scores = np.clip(norm_scores, 0, 1)

    # 4. 计算缩放后的 Patch 大小
    draw_patch_size = int(np.ceil(patch_size_l0[0] / downsample_factor))

    print(f"  -> Drawing {len(coords)} patches manually...")

    for idx, coord in enumerate(coords):
        score = norm_scores[idx]
        color = cmap_obj(score)  # 返回 (r,g,b,a) 0-1
        color_int = tuple([int(c * 255) for c in color[:3]] + [int(255 * alpha)])  # 加入 Alpha

        x, y = coord
        x_draw = int(x / downsample_factor)
        y_draw = int(y / downsample_factor)

        draw.rectangle(
            [x_draw, y_draw, x_draw + draw_patch_size, y_draw + draw_patch_size],
            fill=color_int,
            outline=None
        )

    # 5. 合并并保存
    img.paste(overlay, (0, 0), overlay)
    img.save(save_path)
    print(f"  -> Saved manual heatmap to {save_path}")


# ================= 主程序 =================
parser = argparse.ArgumentParser(description='Heatmap inference script (no YAML required)')
# 以下参数均可选；不传则使用脚本顶部 CONFIG 的默认值
parser.add_argument('--ckpt_dir', type=str, default=None,
                    help='模型权重所在目录（内含 fold-0 ~ fold-N 子目录）')
parser.add_argument('--model_name', type=str, default=None,
                    help='每个 fold 目录下的权重文件名，如 case_best_model.pt')
parser.add_argument('--in_dim', type=int, default=None,
                    help='MambaMIL 输入特征维度（须与训练一致，MUSK=1024）')
parser.add_argument('--patient_list', type=str, default=None,
                    help='要画热图的切片名单 .csv 文件路径（自动读 slide_id 列）；留空则用 CONFIG 中的 patient_list')
parser.add_argument('--output_dir', type=str, default=None,
                    help='统一输出根目录(即 Heatmaps 根)：自动在其下建 production/raw 子目录，CSV 结果落在根目录')
parser.add_argument('--exp_code', type=str, default=None,
                    help='实验名（结果子目录名）')
parser.add_argument('--data_dir', type=str, default=None,
                    help='原始 .tif 切片所在目录')
parser.add_argument('--patch_dir', type=str, default=None,
                    help='patch 坐标 .h5 所在目录')
parser.add_argument('--feature_dir', type=str, default=None,
                    help='预提取特征 .pt 所在目录')
parser.add_argument('--vis_level', type=int, default=None,
                    help='多层级切片的可视化层级')
parser.add_argument('--cmap', type=str, default=None,
                    help='热图颜色映射，如 jet / viridis')
parser.add_argument('--alpha', type=float, default=None,
                    help='热图透明度 0~1')
parser.add_argument('--overlap', type=float, default=None,
                    help='patch 重叠比例')
parser.add_argument('--config_file', type=str, default=None,
                    help='可选的 YAML 配置，用于补充/覆盖 CONFIG')
parser.add_argument('--n_folds', type=int, default=5,
                    help='集成折数（默认 5）')
parser.add_argument('--ensemble', action='store_true', default=True,
                    help='是否使用多折集成（默认开启）')
parser.add_argument('--no_ensemble', dest='ensemble', action='store_false',
                    help='关闭集成，仅用 fold-0')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    config = build_config(args)
    print("\nLoading Configuration...")

    args_ns = argparse.Namespace(**config)
    patch_args = argparse.Namespace(**config['patching_arguments'])
    data_args = argparse.Namespace(**config['data_arguments'])
    model_args = argparse.Namespace(**config['model_arguments'])
    model_args.n_classes = config['exp_arguments']['n_classes']
    exp_args = argparse.Namespace(**config['exp_arguments'])
    heatmap_args = argparse.Namespace(**config['heatmap_arguments'])
    sample_args = argparse.Namespace(**config['sample_arguments'])

    exp_code = config['exp_arguments']['save_exp_code']
    result_dir = config['exp_arguments'].get('result_dir', exp_args.raw_save_dir)
    os.makedirs(result_dir, exist_ok=True)

    patch_size = tuple([patch_args.patch_size] * 2)
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))

    # ---- 确定要处理的切片清单 ----
    patient_ids = load_patient_list(config.get('patient_list') or args.patient_list)
    if patient_ids is not None:
        process_stack = pd.DataFrame({'slide_id': patient_ids})
    elif data_args.process_list:
        csv_path = data_args.process_list
        if not os.path.exists(csv_path):
            csv_path = os.path.join('heatmaps/process_lists', data_args.process_list)
        process_stack = pd.read_csv(csv_path)
    else:
        raise ValueError("Must provide a patient list (CONFIG['patient_list'] / --patient_list) "
                         "or data_arguments.process_list CSV!")

    print('\nInitializing MambaMIL Ensemble...')
    base_dir = model_args.ckpt_dir
    model_name = model_args.model_name
    # 兼容旧写法：若仍提供完整 ckpt_path（含 fold-0/xxx.pt），则从中推导
    if getattr(model_args, 'ckpt_path', None):
        base_dir = os.path.dirname(os.path.dirname(model_args.ckpt_path))
        model_name = os.path.basename(model_args.ckpt_path)

    models_list = []
    folds_to_load = range(args.n_folds) if args.ensemble else [0]

    for fold in folds_to_load:
        ckpt_path_fold = os.path.join(base_dir, f'fold-{fold}', model_name)
        if not os.path.exists(ckpt_path_fold):
            print(f"Warning: Fold {fold} checkpoint not found. Skipping.")
            continue
        print(f"Loading Fold-{fold}: {ckpt_path_fold}")
        # 模型结构必须从 CONFIG / 命令行读，确保与训练时一致
        real_model = MambaMIL(in_dim=model_args.in_dim,
                              n_classes=model_args.n_classes,
                              dropout=model_args.dropout,
                              act=model_args.act,
                              layer=model_args.layer,
                              type=model_args.mamba_type)
        ckpt = torch.load(ckpt_path_fold, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        real_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        real_model.to(device).eval()
        models_list.append(MambaMIL_Heatmap_Wrapper(real_model))

    if len(models_list) == 0: raise ValueError("No models loaded!")
    print(f'MambaMIL Ensemble Loaded! Total models: {len(models_list)}')

    label_dict = data_args.label_dict
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    existing_patch_dir = getattr(data_args, 'existing_patch_dir', None)
    pre_extracted_feature_dir = getattr(data_args, 'pre_extracted_feature_dir', None)

    print(f"\n--- MODE: Ensemble Feature Loading (Smart Heatmap) ---")

    for i in tqdm(range(len(process_stack))):
        csv_slide_id = str(process_stack.loc[i, 'slide_id'])
        slide_id = csv_slide_id

        label = process_stack.loc[i, 'label'] if 'label' in process_stack.columns else 'Unspecified'
        grouping = reverse_label_dict[label] if not isinstance(label, str) else label

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_code, str(grouping))
        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_code, str(grouping), slide_id)
        os.makedirs(p_slide_save_dir, exist_ok=True)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        # 1. 查找文件
        if isinstance(data_args.data_dir, str):
            wsi_base_dir = data_args.data_dir
        else:
            wsi_base_dir = data_args.data_dir[process_stack.loc[i, data_args.data_dir_key]]

        slide_path = generic_locate_file(wsi_base_dir, slide_id, [data_args.slide_ext])
        h5_path = generic_locate_file(existing_patch_dir, slide_id, ['.h5'])
        pt_path = generic_locate_file(pre_extracted_feature_dir, slide_id, ['.pt'])

        print(f'\nProcessing: {slide_id}')
        if not slide_path or not os.path.exists(slide_path):
            print(f"Skipping: Slide not found")
            continue
        if not h5_path or not os.path.exists(h5_path):
            print(f"Skipping: Patch coords not found")
            continue
        if not pt_path or not os.path.exists(pt_path):
            print(f"Skipping: Features not found")
            continue

        # 2. 加载数据
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
        data = torch.load(pt_path, map_location=device)
        features = data['features'] if isinstance(data, dict) else data

        if len(coords) != len(features):
            min_len = min(len(coords), len(features))
            coords = coords[:min_len]
            features = features[:min_len]

        features = features.to(device)
        process_stack.loc[i, 'bag_size'] = len(features)

        # 3. 初始化 WSI
        wsi_object = WholeSlideImage(slide_path)
        wsi_object = inject_dummy_contours(wsi_object)

        # 4. 集成推理
        Y_hats, Y_hats_str, Y_probs, A = infer_ensemble_slide(models_list, features, label, reverse_label_dict,
                                                              exp_args.n_classes)

        # 保存原始 Attention 数据
        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        save_hdf5(block_map_save_path, {'attention_scores': A, 'coords': coords}, mode='w')

        for c in range(exp_args.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]
        process_stack.to_csv(os.path.join(result_dir, f"{exp_code}_results.csv"), index=False)

        # =========================================================================
        # 5. 可视化 (Heatmap) - 【超级修复：分流处理】
        # =========================================================================
        heatmap_save_name = f'{slide_id}_blockmap.png'
        prod_save_path = os.path.join(p_slide_save_dir, heatmap_save_name)

        if not os.path.exists(prod_save_path):
            # 检查层级
            max_level = wsi_object.wsi.level_count - 1

            # --- 分支 A: 正常 Slide (有多层级) ---
            if max_level > 0:
                req_vis_level = min(heatmap_args.vis_level, max_level)
                ref_ds = wsi_object.wsi.level_downsamples[patch_args.patch_level]
                vis_ds = wsi_object.wsi.level_downsamples[req_vis_level]
                scale_factor = (vis_ds / ref_ds)
                vis_patch_size = tuple(
                    (np.array(patch_size) / scale_factor * heatmap_args.custom_downsample).astype(int))

                heatmap_vis_args = {'convert_to_percentiles': True,
                                    'vis_level': req_vis_level,
                                    'blur': heatmap_args.blur,
                                    'custom_downsample': heatmap_args.custom_downsample}

                # 使用 CLAM 原生绘制函数
                heatmap = drawHeatmap(A, coords, slide_path, wsi_object=wsi_object,
                                      cmap=heatmap_args.cmap, alpha=heatmap_args.alpha,
                                      use_holes=False, binarize=False, blank_canvas=False,
                                      thresh=-1, patch_size=vis_patch_size, **heatmap_vis_args)

                os.makedirs(os.path.dirname(prod_save_path), exist_ok=True)
                heatmap.save(prod_save_path)

                # 保存原图叠加 (可选)
                if heatmap_args.save_orig:
                    orig_path = os.path.join(p_slide_save_dir, f'{slide_id}_orig.jpg')
                    if not os.path.exists(orig_path):
                        orig = wsi_object.wsi.read_region((0, 0), req_vis_level,
                                                          wsi_object.wsi.level_dimensions[req_vis_level]).convert('RGB')
                        orig.save(orig_path)

            # --- 分支 B: 异常 Slide (只有1层) - 使用手动绘制避免条纹 ---
            else:
                print(f"  [Warning] Slide only has Level 0. Using Manual Thumbnail Drawing to avoid artifacts.")

                manual_ds_factor = 32

                os.makedirs(os.path.dirname(prod_save_path), exist_ok=True)
                manual_draw_heatmap(wsi_object, coords, A, prod_save_path,
                                    patch_size_l0=patch_size,
                                    downsample_factor=manual_ds_factor,
                                    cmap=heatmap_args.cmap,
                                    alpha=heatmap_args.alpha)

                # 手动保存原图 (缩略图)
                if heatmap_args.save_orig:
                    orig_path = os.path.join(p_slide_save_dir, f'{slide_id}_orig.jpg')
                    if not os.path.exists(orig_path):
                        w, h = wsi_object.wsi.level_dimensions[0]
                        thumb = wsi_object.wsi.get_thumbnail((w // manual_ds_factor, h // manual_ds_factor)).convert(
                            'RGB')
                        thumb.save(orig_path)

        # 6. 采样 (保持不变，采样永远基于 Level 0 坐标)
        for sample in sample_args.samples:
            if sample['sample']:
                tag = f"label_{label}_pred_{Y_hats[0]}"
                sample_save_dir = os.path.join(exp_args.production_save_dir, exp_code,
                                               'sampled_patches', str(tag), sample['name'])
                os.makedirs(sample_save_dir, exist_ok=True)
                sample_results = sample_rois(A, coords, k=sample['k'], mode=sample['mode'], seed=sample.get('seed', 1))
                for idx, (s_coord, s_score) in enumerate(
                        zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, patch_size).convert(
                        'RGB')
                    save_path = os.path.join(sample_save_dir,
                                             f'{idx}_{slide_id}_x_{s_coord[0]}_y_{s_coord[1]}_a_{s_score:.3f}.png')
                    patch.save(save_path)

    print("\nDone! All slides processed.")

    print("\nDone! All slides processed.")

des processed.")

    print("\nDone! All slides processed.")

