"""
训练脚本 - 用于训练医学图像分类模型 (MambaMIL 版本)
支持多fold交叉验证和分层多实例学习 (Slide-Level -> Patient-Level)
"""

import os
import sys
# 获取当前脚本的绝对路径
current_path = os.path.abspath(os.path.dirname(__file__))
# 获取当前脚本的父目录 (项目根目录)
root_path = os.path.split(current_path)[0]
# 将根目录加入 Python 搜索路径
sys.path.append(root_path)

import argparse
import warnings

# 屏蔽警告信息
warnings.filterwarnings("ignore")

# 设置CUDA可见设备 (根据您的服务器情况调整)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

# ============================================
# 导入配置和模型相关模块
# ============================================
# 注意：这里假设您修改后的配置都在 config/config.py 中
from config.config import _C as cfg
from models import build_model
from data import build_data_loader
from engine import (
    Case_predict,
    train_val_case_mamba,
    train_val_case_mamba_DynamicWeight
)
from criterion import (build_criterion)
from optimizer import build_optimizer, build_lr_scheduler
from utils_ import mkdir, setup_logger, seed_torch
from utils_.evaluation import write_pooled_validation_threshold


def main():
    """
    主训练函数
    """
    # -------------------------------------------------------
    # 1. 构建数据加载器
    # -------------------------------------------------------
    loaders = build_data_loader(cfg)

    # -------------------------------------------------------
    # 3. 构建模型
    # -------------------------------------------------------
    # 3.1 构建 Case 级别的预测模型 (ViT/MLP)
    print(f"创建 Case_predict 模型... 模式: {cfg.SOLVER.feat_distillation}")
    case_model = Case_predict(cfg).cuda()

    # 3.2 构建 Slide 级主模型 (这里将调用 build_model 中的 MambaMIL 分支)
    print(f"创建主模型: {cfg.model_type}...")
    model, _ = build_model(cfg, case_model)

    # -------------------------------------------------------
    # 4. 构建优化器 (关键步骤)
    # -------------------------------------------------------
    # 手动打包参数字典，复刻双优化器结构
    # optimizer1 -> 优化 MambaMIL
    # optimizer2 -> 优化 Case_predict
    params = {
        'params': model.parameters(),          # Slide级模型参数
        'lr': cfg.SOLVER.lr,                   # Slide级学习率
        'case_params': case_model.parameters(),# Case级模型参数
        'case_lr': cfg.SOLVER.case_lr,         # Case级学习率
        'epoch': 0
    }

    # 构建两个优化器
    optimizer1, optimizer2 = build_optimizer(cfg, params)

    # 为每个优化器构建学习率调度器
    lr_scheduler1, optimizer1 = build_lr_scheduler(cfg, optimizer1)
    lr_scheduler2, optimizer2 = build_lr_scheduler(cfg, optimizer2)

    # -------------------------------------------------------
    # 5. 构建损失函数
    # -------------------------------------------------------
    criterion = build_criterion(cfg, loaders['train_num_list'])

    # -------------------------------------------------------
    # 6. 开始训练
    # -------------------------------------------------------
    print(f"开始训练 Fold {cfg.DATA.fold_idx}, Seed {cfg.seed}...")

    # 注意：train_val_case_abmil 内部需要处理 model 的返回值。
    # 请确保 MambaMIL.forward 的返回值能被 engine 正确解包。
    train_val_case_mamba_DynamicWeight(
        model, case_model, loaders, criterion,
        lr_scheduler1, lr_scheduler2, optimizer1, optimizer2,
        cfg, params
    )


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练医学图像分类模型 (Mamba版)')
    parser.add_argument(
        "--cfg",
        # 修改默认路径为您新创建的 MambaMIL 配置文件
        default='./configs/MambaMIL_DynamicWeight.yaml',
        metavar='FILE',
        help='配置文件路径',
        type=str
    )
    args = parser.parse_args()

    # 加载配置文件
    cfg.merge_from_file(args.cfg)

    # Use the configured, documented seed for every reported run.
    seed_list = [cfg.seed]

    # 遍历每个种子
    for seed in seed_list:
        seed_torch(seed=seed)

        # 遍历每个fold进行交叉验证
        # 如果只想跑某一个Fold，可以在 yaml 里把 fold_idx 改大，或者在这里手动控制 range
        start_fold = cfg.DATA.fold_idx
        for i in range(start_fold, cfg.DATA.fold_K):
            # 动态更新 config 中的 fold_idx，因为 build_model 里的模型加载路径依赖它
            cfg.DATA.fold_idx = i

            # 构建日志和模型保存路径
            cfg.logs_dir_ = os.path.join(
                cfg.logs_dir, cfg.Task_name, f'seed-{seed}', f'fold-{i}'
            )
            cfg.model_dir_ = os.path.join(
                cfg.model_dir, cfg.Task_name, f'seed-{seed}', f'fold-{i}'
            )

            # 创建目录
            mkdir(cfg.logs_dir_)
            mkdir(cfg.model_dir_)

            # 设置日志记录器
            setup_logger('eval', os.path.join(cfg.logs_dir_, 'eval.log'))
            logger = setup_logger('train', os.path.join(cfg.logs_dir_, 'train.log'))
            logger.info(f"Using Config: \n{cfg}")

            print(f"========== 正在启动 Fold {i} (Model: {cfg.model_type}) ==========")

            # 执行训练
            main()

        experiment_dir = os.path.join(cfg.model_dir, cfg.Task_name, f'seed-{seed}')
        threshold = write_pooled_validation_threshold(experiment_dir, folds=cfg.DATA.fold_K)
        print(f"Pooled validation threshold saved: {threshold:.4f}")
