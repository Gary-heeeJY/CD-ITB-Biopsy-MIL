import torch
from torch.utils.data import DataLoader, SequentialSampler
import logging
import argparse
import os
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# === 引入你的项目模块 ===
current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(current_path)[0]
sys.path.append(root_path)

from config.config import _C as cfg
from data.dataset_generic import Generic_MIL_Dataset, collate_MIL
from utils_ import setup_logger
from models.MambaMIL import MambaMIL
from engine import (
    train_val_case_mamba,
    train_val_case_mamba_DynamicWeight
)
from engine.case_variants import Case_predict_MeanPool
from criterion.metrics import predict_metric, get_biospy_patient_metric, get_patient_feat
from utils_.evaluation import checkpoint_threshold, load_pooled_validation_threshold

# 当前消融变体标签 (A/B/C/D)
VARIANT_TAG = 'A_meanpool'


def resolve_weight_root(cfg):
    """自动探测训练实际保存权重的 seed 目录。

    训练和测试统一使用 seed=421。
    这里不再依赖 cfg.seed，而是直接扫描 model_dir/Task_name 下的 seed-* 目录。
    """
    import glob
    base = os.path.join(cfg.model_dir, cfg.Task_name)
    cands = sorted(glob.glob(os.path.join(base, 'seed-*')))
    if not cands:
        # 兜底：使用配置中的 seed（多数情况下不会走到这里）
        default_root = os.path.join(base, f"seed-{cfg.seed}")
        print(f"[warn] 未在 {base} 下找到 seed-* 目录，回退到 {default_root}")
        return default_root
    # 优先选择包含 fold-0/case_best_model.pt 的目录；否则取修改时间最新的
    for c in cands:
        if os.path.exists(os.path.join(c, 'fold-0', 'case_best_model.pt')):
            return c
    return max(cands, key=os.path.getmtime)


def main(args, device):
    # ==================== 0. 配置覆盖 ====================
    cfg.merge_from_file(args.cfg)
    cfg.DATA.n_classes = 3
    cfg.DATA.case_classes = args.case_classes
    cfg.DATA.hidden_dim = args.embed_dim
    cfg.SOLVER.feat_distillation = args.mode

    # 推导权重 / 结果路径
    # 训练和测试统一使用 seed=421，
    # 因此权重根目录自动探测 model_dir/Task_name 下的真实 seed-* 目录。
    if args.expe_name is None:
        if args.seed is not None:
            args.expe_name = os.path.join(cfg.model_dir, cfg.Task_name, f"seed-{args.seed}")
        else:
            args.expe_name = resolve_weight_root(cfg)
    if args.ex_result is None:
        args.ex_result = os.path.join(cfg.model_dir, cfg.Task_name, "test")
    cfg.logs_dir_ = args.ex_result
    cfg.model_dir_ = args.ex_result

    os.makedirs(args.ex_result, exist_ok=True)
    logger = setup_logger('test_ensemble', os.path.join(args.ex_result, 'test_ensemble_log.txt'))
    logger.info(f"开始 MambaMIL + Case_predict 独立测试 (变体 {VARIANT_TAG})")
    logger.info(f"权重根目录: {args.expe_name}")

    logger.info(f"配置加载完毕，Case Model Depth: {cfg.MODEL.Vit.depth}")

    slide_label_dict = {'CD': 1, 'ITB': 0, 'TB': 0, 'NO': 2}
    case_label_dict = {'CD': 1, 'ITB': 0, 'TB': 0}

    # ==================== 1. 数据加载 ====================
    print(f"Loading data from: {args.label_csv_file}")
    dataset = Generic_MIL_Dataset(csv_path=args.label_csv_file,
                                  data_dir=args.feat_file,
                                  slide_label_dict=slide_label_dict,
                                  case_label_dict=case_label_dict)

    dataloader = DataLoader(dataset, batch_size=1,
                            sampler=SequentialSampler(dataset),
                            collate_fn=collate_MIL, num_workers=2)

    # 建立 Slide -> Case 映射
    slide_to_case_map = {}
    if hasattr(dataset, 'slide_data'):
        df = dataset.slide_data
        slide_to_case_map = dict(zip(df['slide_id'].astype(str), df['case_id'].astype(str)))

    # ==================== 2. 初始化集成存储器 ====================
    ensemble_storage = {}
    validation_thresholds = []
    pooled_threshold = load_pooled_validation_threshold(args.expe_name)
    slide_ensemble_storage = {}

    # ==================== 3. 循环 Folds ====================
    for fold_idx in range(args.folds):
        print(f"\n{'=' * 30}\nProcessing Fold {fold_idx} / {args.folds}\n{'=' * 30}")

        # 初始化指标计算器 (ensure these are fresh for each fold)
        metric_rec1 = predict_metric(cfg=cfg)
        metric_rec1.reset('test')
        case_metric_rec = get_biospy_patient_metric(cfg=cfg)
        case_metric_rec.reset('test')
        patient_info = get_patient_feat(cfg=cfg)
        patient_info.reset('test')

        # --- 构建模型 ---
        model = MambaMIL(
            in_dim=cfg.DATA.hidden_dim,
            n_classes=3,
            dropout=cfg.MODEL.Mamba.dropout,
            act=cfg.MODEL.Mamba.act,
            layer=cfg.MODEL.Mamba.layer,
            rate=cfg.MODEL.Mamba.rate,
            type=cfg.MODEL.Mamba.type
        ).to(device)

        case_model = Case_predict_MeanPool(cfg).to(device)

        # --- 加载权重 ---
        target_model_name = "case_best_model.pt"
        model_path = os.path.join(args.expe_name, f'fold-{fold_idx}', target_model_name)

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        validation_thresholds.append(checkpoint_threshold(checkpoint))

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        if 'case_model' in checkpoint: case_model.load_state_dict(checkpoint['case_model'], strict=False)

        model.eval()
        case_model.eval()

        # ==================== A. Slide 推理 ====================
        with torch.no_grad():
            for batch_id, (feat, label, coords, slide_id, case_label) in enumerate(dataloader):
                feat = feat.to(device)
                label = label.to(device)
                case_label = case_label.to(device)
                s_id = slide_id[0]

                if s_id in slide_to_case_map:
                    case_name = slide_to_case_map[s_id]
                else:
                    case_name = s_id.split('_')[-1][:-2] if '_' in s_id else s_id

                batch_pre = model(feat)
                if isinstance(batch_pre, (list, tuple)):
                    logit = batch_pre[0]
                    slide_feat = batch_pre[1]
                else:
                    logit = batch_pre
                    slide_feat = None

                if logit.dim() == 3: logit = logit.squeeze(1)

                # 更新 Slide 指标
                metric_rec1.update(logit, label, slide_id)
                # 存储 Patient 信息
                patient_info.update(case_name, slide_feat.detach(), case_label, logit, label)

                # Slide 集成存储
                slide_prob = torch.nn.functional.softmax(logit, dim=-1).cpu().numpy()[0]
                if s_id not in slide_ensemble_storage:
                    slide_ensemble_storage[s_id] = {'prob_sum': np.zeros_like(slide_prob), 'label': label.item(),
                                                    'case_id': case_name, 'count': 0}
                slide_ensemble_storage[s_id]['prob_sum'] += slide_prob
                slide_ensemble_storage[s_id]['count'] += 1

        # ==================== B. Case 推理 ====================
        case_info_all = patient_info.get_value
        case_predictions_detail = []

        with torch.no_grad():
            for case_name, case_info in case_info_all.items():
                case_feat = case_info['case_feat']
                case_label_t = case_info['case_label']
                slide_logits = case_info['logit']
                if slide_logits.dim() == 3: slide_logits = slide_logits.squeeze(1)

                case_dict = case_model(case_feat, slide_logits)
                case_logit = case_dict['case_logit']

                # 更新 Case 指标
                case_metric_rec.update(case_logit, case_label_t, case_name)

                case_prob = torch.nn.functional.softmax(case_logit, dim=-1).cpu().numpy()[0]
                pred_cls = np.argmax(case_prob)

                # 记录本折预测
                case_predictions_detail.append({
                    'CaseID': case_name,
                    'Label': case_label_t.item(),
                    'Prob_CD': case_prob[1],
                    'Prob_TB': case_prob[0],
                    'Pred': pred_cls
                })

                # Case 集成存储
                if case_name not in ensemble_storage:
                    ensemble_storage[case_name] = {'prob_sum': np.zeros_like(case_prob), 'label': case_label_t.item(),
                                                   'count': 0}
                ensemble_storage[case_name]['prob_sum'] += case_prob
                ensemble_storage[case_name]['count'] += 1

        # 保存每一折的预测结果
        if args.save_detail_result:
            df_fold = pd.DataFrame(case_predictions_detail)
            df_fold.to_csv(os.path.join(args.ex_result, f'fold_{fold_idx}_predictions.csv'), index=False)

        # 【核心修改】正确打印每一折的指标 (利用对象的 __str__ 方法)
        logger.info(f"Fold:{fold_idx} Finished.")
        logger.info(f" Slide: {metric_rec1}")
        logger.info(f" Case: {case_metric_rec}")

    # ==================== 4. 最终集成结果 ====================
    print(f"\n{'=' * 30}\nCalculating Ensemble Results\n{'=' * 30}")

    # --- Case Level (Case_Final_Results) ---
    ensemble_results = []
    y_true, y_scores_pos, y_pred = [], [], []

    for case_name, data in ensemble_storage.items():
        if data['count'] == 0: continue
        avg_prob = data['prob_sum'] / data['count']

        y_true.append(data['label'])
        y_scores_pos.append(avg_prob[1])

        ensemble_results.append({
            'CaseID': case_name, 'Label': data['label'],
            'Avg_Prob_CD': avg_prob[1], 'Avg_Prob_TB': avg_prob[0],
            'Ensemble_Pred': 0
        })

    y_true = np.array(y_true)
    y_scores_pos = np.array(y_scores_pos)

    final_threshold = (
        float(pooled_threshold)
        if pooled_threshold is not None
        else (float(np.mean(validation_thresholds)) if validation_thresholds else 0.5)
    )
    source = "pooled out-of-fold validation" if pooled_threshold is not None else "legacy fold average"
    logger.info(f"Case threshold from {source}: {final_threshold:.4f}")

    # 更新预测值和结果列表
    y_pred = (y_scores_pos > final_threshold).astype(int)
    for i, res in enumerate(ensemble_results):
        res['Ensemble_Pred'] = y_pred[i]

    # 打印最终 Case 指标
    logger.info("--- Case Ensemble Metrics ---")
    logger.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    try:
        logger.info(f"AUC: {roc_auc_score(y_true, y_scores_pos):.4f}")
    except:
        pass
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    logger.info(f"Report:\n{classification_report(y_true, y_pred, target_names=['ITB', 'CD'], digits=4)}")

    pd.DataFrame(ensemble_results).to_csv(os.path.join(args.ex_result, 'Case_Final_Results.csv'), index=False)

    # --- Slide Level (Slide_Final_Results) ---
    slide_results = []
    y_true_s, y_prob_s, y_pred_s = [], [], []

    for s_id, data in slide_ensemble_storage.items():
        if data['count'] == 0: continue
        avg_prob = data['prob_sum'] / data['count']
        pred = np.argmax(avg_prob)

        y_true_s.append(data['label'])
        y_pred_s.append(pred)
        y_prob_s.append(avg_prob)

        slide_results.append({
            'SlideID': s_id, 'CaseID': data['case_id'], 'Label': data['label'],
            'Avg_Prob_CD': avg_prob[1], 'Avg_Prob_TB': avg_prob[0], 'Avg_Prob_NO': avg_prob[2],
            'Ensemble_Pred': pred
        })

    # 打印最终 Slide 指标
    logger.info("--- Slide Ensemble Metrics ---")
    logger.info(f"Accuracy: {accuracy_score(y_true_s, y_pred_s):.4f}")
    try:
        auc_s = roc_auc_score(y_true_s, np.array(y_prob_s), multi_class='ovr')
        logger.info(f"AUC (OVR): {auc_s:.4f}")
    except:
        pass

    pd.DataFrame(slide_results).to_csv(os.path.join(args.ex_result, 'Slide_Final_Results.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径配置 (请确认这些默认路径是否正确)
    parser.add_argument('--cfg', type=str,
                        default='./configs/MambaMIL_DynamicWeight_A_meanpool.yaml')
    parser.add_argument('--ex_csv', type=str, default='./data/splits/internal_test.csv')
    parser.add_argument('--label_csv_file', type=str,
                        default='./data/labels/internal_test.csv')
    parser.add_argument('--expe_name', type=str, default=None)
    parser.add_argument('--folds', type=int, default=5)
    # 显式指定测试权重对应的 seed；默认自动探测（训练实际用 421）
    parser.add_argument('--seed', type=int, default=None,
                        help='测试权重对应的 seed；默认自动探测 model_dir/Task_name 下的 seed-* 目录')
    parser.add_argument('--ex_result', type=str,
                        default=None)
    parser.add_argument('--feat_file', type=str,
                        default='./data/features/MUSK')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--case_classes', type=int, default=2)
    parser.add_argument('--mode', type=str, default='Mean')
    parser.add_argument('--save_detail_result', type=bool, default=True)

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    main(args, device)
