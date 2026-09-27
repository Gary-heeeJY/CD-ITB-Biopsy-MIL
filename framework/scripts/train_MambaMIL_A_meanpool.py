"""
训练脚本 - 【消融实验 A】Mean pooling 变体
================================================
Slide: MambaMIL | AFS: × | Transformer: × | 聚合: mean(slide features) -> MLP
完整模型 (变体 D) 见 train_MambaMIL.py。
"""
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]
sys.path.append(root_path)

import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

from config.config import _C as cfg
from models import build_model
from data import build_data_loader
from engine import (
    train_val_case_mamba,
    train_val_case_mamba_DynamicWeight
)
from engine.case_variants import Case_predict_MeanPool
from criterion import (build_criterion)
from optimizer import build_optimizer, build_lr_scheduler
from utils_ import mkdir, setup_logger, seed_torch
from utils_.evaluation import write_pooled_validation_threshold


VARIANT = 'A'  # Mean pooling: 无AFS, 无Transformer


def main():
    print(f"[消融-{VARIANT}] Case_predict 变体: Mean pooling (无 AFS, 无 Transformer)")

    run_name = f"{cfg.Task_name}_seed{cfg.seed}_fold{cfg.DATA.fold_idx}"

    loaders = build_data_loader(cfg)

    case_model = Case_predict_MeanPool(cfg).cuda()
    print(f"创建主模型: {cfg.model_type}...")
    model, _ = build_model(cfg, case_model)

    params = {
        'params': model.parameters(),
        'lr': cfg.SOLVER.lr,
        'case_params': case_model.parameters(),
        'case_lr': cfg.SOLVER.case_lr,
        'epoch': 0
    }
    optimizer1, optimizer2 = build_optimizer(cfg, params)
    lr_scheduler1, optimizer1 = build_lr_scheduler(cfg, optimizer1)
    lr_scheduler2, optimizer2 = build_lr_scheduler(cfg, optimizer2)

    criterion = build_criterion(cfg, loaders['train_num_list'])

    print(f"开始训练 Fold {cfg.DATA.fold_idx}, Seed {cfg.seed}...")
    train_val_case_mamba_DynamicWeight(
        model, case_model, loaders, criterion,
        lr_scheduler1, lr_scheduler2, optimizer1, optimizer2,
        cfg, params
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[消融-A] MambaMIL + Mean pooling')
    parser.add_argument(
        "--cfg",
         default='./configs/MambaMIL_DynamicWeight_A_meanpool.yaml',
        metavar='FILE',
        help='配置文件路径',
        type=str
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)

    seed_list = [cfg.seed]
    for seed in seed_list:
        seed_torch(seed=seed)
        start_fold = cfg.DATA.fold_idx
        for i in range(start_fold, cfg.DATA.fold_K):
            cfg.DATA.fold_idx = i
            cfg.logs_dir_ = os.path.join(cfg.logs_dir, cfg.Task_name, f'seed-{seed}', f'fold-{i}')
            cfg.model_dir_ = os.path.join(cfg.model_dir, cfg.Task_name, f'seed-{seed}', f'fold-{i}')
            mkdir(cfg.logs_dir_)
            mkdir(cfg.model_dir_)
            setup_logger('eval', os.path.join(cfg.logs_dir_, 'eval.log'))
            logger = setup_logger('train', os.path.join(cfg.logs_dir_, 'train.log'))
            logger.info(f"Using Config: \n{cfg}")
            print(f"========== [消融-{VARIANT}] 正在启动 Fold {i} (Model: {cfg.model_type}) ==========")
            main()
        experiment_dir = os.path.join(cfg.model_dir, cfg.Task_name, f'seed-{seed}')
        threshold = write_pooled_validation_threshold(experiment_dir, folds=cfg.DATA.fold_K)
        print(f"Pooled validation threshold saved: {threshold:.4f}")
