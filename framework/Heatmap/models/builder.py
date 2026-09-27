import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from timm.models import create_model
from .timm_wrapper import TimmCNNEncoder
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms

# 尝试导入 safetensors，如果没有则提示安装
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    print("Warning: 'safetensors' library not found. If using .safetensors models, please run: pip install safetensors")

# ================= MUSK Extension =================
current_models_dir = os.path.dirname(os.path.abspath(__file__))
clam_root = os.path.dirname(current_models_dir)
musk_root = os.path.join(clam_root, 'MUSK')

if musk_root not in sys.path:
    sys.path.append(musk_root)

try:
    from MUSK_model import modeling
except ImportError:
    pass


class MuskEncoderWrapper(nn.Module):
    def __init__(self, musk_model):
        super().__init__()
        self.model = musk_model

    def forward(self, x):
        img_features, _ = self.model(
            image=x,
            return_global=True,
            with_head=False,
            out_norm=False,
            ms_aug=False
        )
        return img_features


def get_musk_encoder(ckpt_path, target_img_size=384):
    print(f'\nInitializing MUSK encoder from: {ckpt_path}')

    # 1. 创建模型架构
    try:
        model = create_model("musk_large_patch16_384", pretrained=False)
    except Exception as e:
        print(f"Error creating MUSK model: {e}")
        sys.exit(1)

    if not os.path.exists(ckpt_path):
        print(f"Error: MUSK checkpoint not found at {ckpt_path}")
        sys.exit(1)

    # 2. 智能加载权重 (区分 .pt/.pth 和 .safetensors)
    if ckpt_path.endswith('.safetensors'):
        print("Detected safetensors format, using load_file...")
        state_dict = load_safetensors(ckpt_path, device='cpu')
    else:
        try:
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location='cpu')

    # 3. 权重 Key 清洗
    # 有些 checkpoint 会包裹在 'model' 或 'module' 键下
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'module' in state_dict:
        state_dict = state_dict['module']

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        # 去除前缀，匹配 timm 模型定义
        if name.startswith('module.'): name = name[7:]
        if name.startswith('musk.'): name = name[5:]
        new_state_dict[name] = v

    # 4. 加载进模型
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"MUSK Weights Loaded. Missing keys (text parts usually): {len(msg.missing_keys)}")

    # 5. 包装与预处理
    wrapped_model = MuskEncoderWrapper(model)
    wrapped_model.eval()

    # 严格对齐训练时的预处理：Resize(384) -> ToTensor -> Normalize(0.5)
    img_transforms = transforms.Compose([
        transforms.Resize((target_img_size, target_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return wrapped_model, img_transforms


def get_encoder(model_name, target_img_size=224):
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size=target_img_size)
    return model, img_transforms