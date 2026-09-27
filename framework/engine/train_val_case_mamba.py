import os
import logging
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils_.logger import Losslogger, Losslogger1
from utils_.evaluation import save_validation_predictions, youden_threshold


# 辅助函数: 获取 losslogger 平均值
def get_avg(logger, key):
    return logger.GetValue(key, 'GetAverageValue')


Losslogger.avg = get_avg
Losslogger1.avg = get_avg


def train_val_case_mamba(model, case_model, loaders, creterion, lr_scheduler1, lr_scheduler2, optimizer1, optimizer2,
                         cfg, params):
    """
    基础版训练函数：标准 Slide Loss (所有 Slide 都参与计算)
    """
    logger = logging.getLogger('train')
    n_epoch = cfg.SOLVER.num_epochs

    loss_func = creterion['slide_loss']
    case_loss_func = creterion['case_loss']
    patient_info = creterion['case_info']

    # --- 初始化三个指标的记录器 ---
    best_result_slide = float("-inf")
    best_result_case = float("-inf")
    best_result_joint = float("-inf")

    for epoch in range(params['epoch'], n_epoch):
        phases = ['train', 'eval'] if epoch % cfg.SOLVER.val_step == 0 else ['train']

        for phase in phases:
            loader = loaders[phase]
            loss_rec = Losslogger(delimiter=' ')
            case_loss_rec = Losslogger1(delimiter=' ')
            patient_info.reset(phase)

            # 初始化列表
            epoch_slide_probs, epoch_slide_labels, epoch_slide_preds = [], [], []
            epoch_case_probs, epoch_case_labels, epoch_case_preds = [], [], []

            if phase == 'train':
                model.train()
                case_model.train()
                current_lr = lr_scheduler1.get_lr()
                logger.info(f'Learning rate: {current_lr}')
            else:
                model.eval()
                case_model.eval()

            # --- Phase 1: Slide Level ---
            for batch_id, (feat, label, coords, slide_id, case_label) in enumerate(loader):
                feat = feat.to(cfg.gpu_id)
                label = label.to(cfg.gpu_id)
                case_label = case_label.to(cfg.gpu_id)
                case_name = slide_id[0].split('_')[-1][:-2] if '_' in slide_id[0] else slide_id[0]

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(feat)

                    if isinstance(output, (tuple, list)):
                        logit = output[0]
                        slide_feat = output[1]
                    else:
                        logit = output
                        slide_feat = None

                    loss_dict = loss_func(logit, label)
                    loss_total = loss_dict['loss']

                    patient_info.update(case_name, slide_feat.detach(), case_label, logit.detach(), label)

                    prob = F.softmax(logit, dim=1).detach().cpu().numpy()
                    pred = torch.argmax(logit, dim=1).detach().cpu().numpy()
                    epoch_slide_probs.extend(prob)
                    epoch_slide_preds.extend(pred)
                    epoch_slide_labels.extend(label.detach().cpu().numpy())

                if phase == 'train':
                    optimizer1.zero_grad()
                    loss_total.backward()
                    optimizer1.step()
                    loss_rec.updata(**loss_dict)

            # Slide Metrics
            slide_acc = accuracy_score(epoch_slide_labels, epoch_slide_preds)
            try:
                if cfg.DATA.n_classes == 2:
                    slide_auc = roc_auc_score(epoch_slide_labels, np.array(epoch_slide_probs)[:, 1])
                else:
                    slide_auc = roc_auc_score(epoch_slide_labels, epoch_slide_probs, multi_class='ovr')
            except:
                slide_auc = 0.5

            # --- Phase 2: Case Level ---
            case_info_all = patient_info.get_value
            with torch.set_grad_enabled(phase == 'train'):
                for case_name in list(case_info_all.keys()):
                    case_data = case_info_all[case_name]

                    case_dict = case_model(case_data['case_feat'], case_data['logit'])
                    case_logit = case_dict['case_logit']

                    case_loss = case_loss_func(case_logit, case_data['case_label'])

                    prob = F.softmax(case_logit, dim=1).detach().cpu().numpy()
                    pred = torch.argmax(case_logit, dim=1).detach().cpu().numpy()
                    epoch_case_probs.extend(prob)
                    epoch_case_preds.extend(pred)
                    epoch_case_labels.extend(case_data['case_label'].detach().cpu().numpy())

                    if phase == 'train':
                        optimizer2.zero_grad()
                        case_loss['case_loss'].backward()
                        optimizer2.step()
                        case_loss_rec.updata(**case_loss)

            # Case Metrics
            case_acc = accuracy_score(epoch_case_labels, epoch_case_preds)
            try:
                case_auc = roc_auc_score(epoch_case_labels, np.array(epoch_case_probs)[:, 1])
            except:
                case_auc = 0.5
            case_threshold = youden_threshold(epoch_case_labels, np.array(epoch_case_probs)[:, 1]) if phase == 'eval' else 0.5

            # Log & Save
            print(
                f"[{phase}] Ep:{epoch} | Slide AUC:{slide_auc:.4f} Loss:{loss_rec.avg('loss'):.3f} | Case AUC:{case_auc:.4f}")

            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'case_model': case_model.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'case_threshold': case_threshold,
            }

            if phase == 'eval' and slide_auc > best_result_slide:
                best_result_slide = slide_auc
                torch.save(state, os.path.join(cfg.model_dir_, "slide_best_model.pt"))
                print(f"  >>> Saved Best Slide: {best_result_slide:.4f}")

            if phase == 'eval' and case_auc > best_result_case:
                best_result_case = case_auc
                torch.save(state, os.path.join(cfg.model_dir_, "case_best_model.pt"))
                save_validation_predictions(
                    cfg.model_dir_, epoch_case_labels, np.asarray(epoch_case_probs)[:, 1]
                )
                print(f"  >>> Saved Best Case: {best_result_case:.4f}")

            current_score = 0.5 * slide_auc + 0.5 * case_auc
            if phase == 'eval' and current_score > best_result_joint:
                best_result_joint = current_score
                torch.save(state, os.path.join(cfg.model_dir_, "joint_best_model.pt"))
                print(f"  >>> Saved Best Joint: {current_score:.4f} (S:{slide_auc:.2f}+C:{case_auc:.2f})")

        lr_scheduler1.step()
        lr_scheduler2.step()
    logger.info('Training Finished!')

def train_val_case_mamba_accum(model, case_model, loaders, creterion, lr_scheduler1, lr_scheduler2, optimizer1,
                               optimizer2, cfg, params):
    """
    进阶版训练函数：梯度累积 + 联合保存
    """
    logger = logging.getLogger('train')
    n_epoch = cfg.SOLVER.num_epochs

    loss_func = creterion['slide_loss']
    case_loss_func = creterion['case_loss']
    patient_info = creterion['case_info']

    best_slide_auc = 0.0
    best_case_auc = 0.0
    best_joint_score = 0.0

    ACCUMULATION_STEPS = 32
    CASE_ACCUM_STEPS = 4

    for epoch in range(params['epoch'], n_epoch):
        phases = ['train', 'eval'] if epoch % cfg.SOLVER.val_step == 0 else ['train']

        for phase in phases:
            loader = loaders[phase]
            loss_rec = Losslogger(delimiter=' ')
            case_loss_rec = Losslogger1(delimiter=' ')
            patient_info.reset(phase)

            slide_probs_list, slide_labels_list, slide_preds_list = [], [], []
            case_probs_list, case_labels_list, case_preds_list = [], [], []

            if phase == 'train':
                model.train()
                case_model.train()
                current_lr = lr_scheduler1.get_lr()
                logger.info(f'Learning rate: {current_lr}')
                optimizer1.zero_grad()
                optimizer2.zero_grad()
            else:
                model.eval()
                case_model.eval()

            # --- Phase 1: Slide Level (Accum) ---
            total_step = len(loader)
            for batch_id, (feat, label, coords, slide_id, case_label) in enumerate(loader):
                feat = feat.to(cfg.gpu_id)
                label = label.to(cfg.gpu_id)
                case_label = case_label.to(cfg.gpu_id)
                case_name = slide_id[0].split('_')[-1][:-2] if '_' in slide_id[0] else slide_id[0]

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(feat)
                    if isinstance(output, (tuple, list)):
                        logit = output[0]
                        slide_feat = output[1]
                    else:
                        logit = output
                        slide_feat = None

                    loss_dict = loss_func(logit, label)
                    loss_total = loss_dict['loss']

                    loss_for_backward = loss_total / ACCUMULATION_STEPS

                    patient_info.update(case_name, slide_feat.detach(), case_label, logit.detach(), label)

                    prob = F.softmax(logit, dim=1).detach().cpu().numpy()
                    pred = torch.argmax(logit, dim=1).detach().cpu().numpy()
                    slide_probs_list.extend(prob)
                    slide_preds_list.extend(pred)
                    slide_labels_list.extend(label.detach().cpu().numpy())

                if phase == 'train':
                    loss_for_backward.backward()
                    if (batch_id + 1) % ACCUMULATION_STEPS == 0 or (batch_id + 1) == total_step:
                        optimizer1.step()
                        optimizer1.zero_grad()
                    loss_rec.updata(**loss_dict)

            # Slide Metrics
            slide_acc = accuracy_score(slide_labels_list, slide_preds_list)
            try:
                if cfg.DATA.n_classes == 2:
                    slide_auc = roc_auc_score(slide_labels_list, np.array(slide_probs_list)[:, 1])
                else:
                    slide_auc = roc_auc_score(slide_labels_list, slide_probs_list, multi_class='ovr')
            except:
                slide_auc = 0.5

            # --- Phase 2: Case Level (Accum) ---
            case_info_all = patient_info.get_value
            if phase == 'train': optimizer2.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                for i, case_name in enumerate(list(case_info_all.keys())):
                    case_data = case_info_all[case_name]

                    case_dict = case_model(case_data['case_feat'], case_data['logit'])
                    case_logit = case_dict['case_logit']

                    case_loss = case_loss_func(case_logit, case_data['case_label'])
                    loss_backward = case_loss['case_loss'] / CASE_ACCUM_STEPS

                    prob = F.softmax(case_logit, dim=1).detach().cpu().numpy()
                    pred = torch.argmax(case_logit, dim=1).detach().cpu().numpy()
                    case_probs_list.extend(prob)
                    case_preds_list.extend(pred)
                    case_labels_list.extend(case_data['case_label'].detach().cpu().numpy())

                    if phase == 'train':
                        loss_backward.backward()
                        case_loss_rec.updata(**case_loss)
                        if (i + 1) % CASE_ACCUM_STEPS == 0 or (i + 1) == len(case_info_all):
                            optimizer2.step()
                            optimizer2.zero_grad()

            # Case Metrics
            case_acc = accuracy_score(case_labels_list, case_preds_list)
            try:
                case_auc = roc_auc_score(case_labels_list, np.array(case_probs_list)[:, 1])
            except:
                case_auc = 0.5

            # Log & Save
            print(
                f"[{phase}] Ep:{epoch} | Slide AUC:{slide_auc:.4f} Loss:{loss_rec.avg('loss'):.3f} | Case AUC:{case_auc:.4f}")

            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'case_model': case_model.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict()
            }

            if phase == 'eval' and slide_auc > best_slide_auc:
                best_slide_auc = slide_auc
                torch.save(state, os.path.join(cfg.model_dir_, "slide_best_model.pt"))
                print(f"  >>> Saved Best Slide: {best_slide_auc:.4f}")

            if phase == 'eval' and case_auc > best_case_auc:
                best_case_auc = case_auc
                torch.save(state, os.path.join(cfg.model_dir_, "case_best_model.pt"))
                save_validation_predictions(
                    cfg.model_dir_, case_labels_list, np.asarray(case_probs_list)[:, 1]
                )
                print(f"  >>> Saved Best Case: {best_case_auc:.4f}")

            current_score = 0.5 * slide_auc + 0.5 * case_auc
            if phase == 'eval' and current_score > best_joint_score:
                best_joint_score = current_score
                torch.save(state, os.path.join(cfg.model_dir_, "joint_best_model.pt"))
                print(f"  >>> Saved Best Joint: {current_score:.4f} (S:{slide_auc:.2f}+C:{case_auc:.2f})")

        lr_scheduler1.step()
        lr_scheduler2.step()

    logger.info('Training Finished!')


def train_val_case_mamba_DynamicWeight(model, case_model, loaders, creterion, lr_scheduler1, lr_scheduler2, optimizer1,
                                       optimizer2,
                                       cfg, params):
    """
    【新策略】动态权重训练函数
    1. 移除 Feature Jittering，回归纯净训练。
    2. 引入动态 Loss 权重：Slide Loss 随 Epoch 衰减，Case Loss 随 Epoch 增加。
       目的：前期学特征，后期冲指标，解决验证集震荡问题。
    """
    logger = logging.getLogger('train')
    n_epoch = cfg.SOLVER.num_epochs

    loss_func = creterion['slide_loss']
    case_loss_func = creterion['case_loss']
    patient_info = creterion['case_info']

    # --- 初始化指标 ---
    best_result_slide = float("-inf")
    best_result_case = float("-inf")
    best_result_joint = float("-inf")

    # ============================================================
    # 定义动态权重计算函数
    # ============================================================
    def get_dynamic_weights(current_epoch, total_epochs):
        # 进度 (0.0 ~ 1.0)
        progress = current_epoch / total_epochs

        # Slide Loss 权重: 1.0 -> 0.1 (前期重要，后期辅助)
        w_slide = 1.0 - 0.9 * progress

        # Case Loss 权重: 1.0 -> 2.0 (后期主导，冲刺 AUC)
        w_case = 1.0 + 1.0 * progress

        return w_slide, w_case

    for epoch in range(params['epoch'], n_epoch):
        phases = ['train', 'eval'] if epoch % cfg.SOLVER.val_step == 0 else ['train']

        # 获取当前 Epoch 的权重
        w_slide, w_case = get_dynamic_weights(epoch, n_epoch)
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Slide Weight={w_slide:.2f}, Case Weight={w_case:.2f}")

        for phase in phases:
            loader = loaders[phase]
            loss_rec = Losslogger(delimiter=' ')
            case_loss_rec = Losslogger1(delimiter=' ')
            patient_info.reset(phase)

            epoch_slide_probs, epoch_slide_labels, epoch_slide_preds = [], [], []
            epoch_case_probs, epoch_case_labels, epoch_case_preds = [], [], []

            if phase == 'train':
                model.train()
                case_model.train()
                current_lr = lr_scheduler1.get_lr()
                logger.info(f'Learning rate: {current_lr}')
            else:
                model.eval()
                case_model.eval()

            # --- Phase 1: Slide Level ---
            for batch_id, (feat, label, coords, slide_id, case_label) in enumerate(loader):
                feat = feat.to(cfg.gpu_id)
                label = label.to(cfg.gpu_id)
                case_label = case_label.to(cfg.gpu_id)
                case_name = slide_id[0].split('_')[-1][:-2] if '_' in slide_id[0] else slide_id[0]

                # 【移除】 Feature Jittering 被移除，保持数据纯净

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(feat)

                    if isinstance(output, (tuple, list)):
                        logit = output[0]
                        slide_feat = output[1]
                    else:
                        logit = output
                        slide_feat = None

                    # 计算 Slide Loss
                    loss_output = loss_func(logit, label)

                    if isinstance(loss_output, dict):
                        loss_slide_val = loss_output['loss']
                        loss_dict = loss_output
                    else:
                        loss_slide_val = loss_output
                        loss_dict = {'loss': loss_output}

                    # 【关键】应用动态权重 (虽然这里只算 Slide Loss，但在反向传播时我们会乘以权重)
                    # 注意：这里我们先不算总 Loss，等 Phase 2 算完 Case Loss 后再一起 backward
                    # 或者 (更简单的做法) Phase 1 和 Phase 2 分开优化，分别乘权重

                    # 方案：分开优化 (你的原代码结构就是分开的)
                    # 所以直接在 backward 时乘以权重即可
                    loss_total = loss_slide_val * w_slide

                    patient_info.update(case_name, slide_feat.detach(), case_label, logit.detach(), label)

                    prob = F.softmax(logit, dim=1).detach().cpu().numpy()
                    pred = torch.argmax(logit, dim=1).detach().cpu().numpy()
                    epoch_slide_probs.extend(prob)
                    epoch_slide_preds.extend(pred)
                    epoch_slide_labels.extend(label.detach().cpu().numpy())

                if phase == 'train':
                    optimizer1.zero_grad()
                    loss_total.backward()  # 已乘权重
                    optimizer1.step()
                    loss_rec.updata(**loss_dict)

            # Slide Metrics
            slide_acc = accuracy_score(epoch_slide_labels, epoch_slide_preds)
            try:
                if cfg.DATA.n_classes == 2:
                    slide_auc = roc_auc_score(epoch_slide_labels, np.array(epoch_slide_probs)[:, 1])
                else:
                    slide_auc = roc_auc_score(epoch_slide_labels, epoch_slide_probs, multi_class='ovr')
            except:
                slide_auc = 0.5

            # --- Phase 2: Case Level ---
            case_info_all = patient_info.get_value
            with torch.set_grad_enabled(phase == 'train'):
                for case_name in list(case_info_all.keys()):
                    case_data = case_info_all[case_name]

                    case_dict = case_model(case_data['case_feat'], case_data['logit'])
                    case_logit = case_dict['case_logit']

                    case_loss_dict = case_loss_func(case_logit, case_data['case_label'])
                    case_loss_val = case_loss_dict['case_loss']

                    # 【关键】应用动态权重
                    loss_case_total = case_loss_val * w_case

                    prob = F.softmax(case_logit, dim=1).detach().cpu().numpy()
                    pred = torch.argmax(case_logit, dim=1).detach().cpu().numpy()
                    epoch_case_probs.extend(prob)
                    epoch_case_preds.extend(pred)
                    epoch_case_labels.extend(case_data['case_label'].detach().cpu().numpy())

                    if phase == 'train':
                        optimizer2.zero_grad()
                        loss_case_total.backward()  # 已乘权重
                        optimizer2.step()
                        case_loss_rec.updata(**case_loss_dict)

            # Case Metrics
            case_acc = accuracy_score(epoch_case_labels, epoch_case_preds)
            try:
                case_auc = roc_auc_score(epoch_case_labels, np.array(epoch_case_probs)[:, 1])
            except:
                case_auc = 0.5
            case_threshold = youden_threshold(epoch_case_labels, np.array(epoch_case_probs)[:, 1]) if phase == 'eval' else 0.5

            # Log & Save
            print(
                f"[{phase}] Ep:{epoch} | Slide AUC:{slide_auc:.4f} Loss:{loss_rec.avg('loss'):.3f} | Case AUC:{case_auc:.4f}")

            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'case_model': case_model.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'case_threshold': case_threshold,
            }

            if phase == 'eval' and slide_auc > best_result_slide:
                best_result_slide = slide_auc
                torch.save(state, os.path.join(cfg.model_dir_, "slide_best_model.pt"))
                print(f"  >>> Saved Best Slide: {best_result_slide:.4f}")

            if phase == 'eval' and case_auc > best_result_case:
                best_result_case = case_auc
                torch.save(state, os.path.join(cfg.model_dir_, "case_best_model.pt"))
                save_validation_predictions(
                    cfg.model_dir_, epoch_case_labels, np.asarray(epoch_case_probs)[:, 1]
                )
                print(f"  >>> Saved Best Case: {best_result_case:.4f}")

            current_score = 0.5 * slide_auc + 0.5 * case_auc
            if phase == 'eval' and current_score > best_result_joint:
                best_result_joint = current_score
                torch.save(state, os.path.join(cfg.model_dir_, "joint_best_model.pt"))
                print(f"  >>> Saved Best Joint: {current_score:.4f} (S:{slide_auc:.2f}+C:{case_auc:.2f})")

        lr_scheduler1.step()
        lr_scheduler2.step()
    logger.info('Training Finished!')
