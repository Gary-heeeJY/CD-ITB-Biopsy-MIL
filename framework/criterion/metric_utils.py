import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
from sklearn.metrics import roc_curve, auc

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    # SR = SR.view(-1)
    # GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # GT = GT == np.max(GT)
    corr = torch.sum(SR == GT)
    # corr = np.sum(SR == GT)
    # tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr) / float(SR.size(0))

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # SR = SR.view(-1)
    # GT = GT.view(-1)
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # GT = GT == np.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1)*1 + (GT == 1)*1) == 2)*1
    FN = (((SR == 0)*1 + (GT == 1)*1) == 2)*1

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    # SR = SR.view(-1)
    # GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # GT = GT == np.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0)*1 + (GT == 0)*1) == 2)*1
    FP = (((SR == 1)*1 + (GT == 0)*1) == 2)*1

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    # SR = SR.view(-1)
    # GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # GT = GT == np.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = (((SR == 1)*1 + (GT == 1)*1) == 2)*1
    FP = (((SR == 1)*1 + (GT == 0)*1) == 2)*1

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC

def get_case_AUC(SR, GT):
    pred_value = SR
    true_label = GT

    if (sum(true_label) == len(pred_value)) or (sum(true_label) == 0):
        AUC = 0
        print('only one class')
    else:
        AUC = metrics.roc_auc_score(true_label, pred_value)

    return AUC

def get_AUC(SR, GT):
    # SR = SR.view(-1)
    # GT = GT.view(-1)
    pred_value = SR.tolist() #.data.cpu().numpy()
    true_label = GT.tolist() #.data.cpu().numpy()

    if (sum(true_label) == len(pred_value)) or (sum(true_label) == 0):
        AUC = 0
        print('only one class')
    else:
        AUC = metrics.roc_auc_score(true_label, pred_value)

    return AUC


def get_best_case_threshold(SR, GT):
    """计算最佳阈值，添加边界情况处理"""
    try:
        # 转换为numpy数组
        pred_value = SR.data.cpu().numpy() if torch.is_tensor(SR) else np.array(SR)
        true_label = GT.data.cpu().numpy() if torch.is_tensor(GT) else np.array(GT)

        # 检查输入有效性
        if len(pred_value) == 0 or len(true_label) == 0:
            print("警告: 概率或标签为空，返回默认阈值0.5")
            return 0.5

        # 检查标签是否只有一类
        unique_labels = np.unique(true_label)
        if len(unique_labels) < 2:
            print(f"警告: 标签只有{len(unique_labels)}类，返回默认阈值0.5")
            return 0.5

        # 计算ROC曲线
        fpr, tpr, thresholds = metrics.roc_curve(true_label, pred_value)

        # 检查阈值是否有效
        if len(thresholds) == 0:
            print("警告: 无法计算ROC阈值，返回默认阈值0.5")
            return 0.5

        # 计算Youden指数
        Youden_index = tpr + (1 - fpr)

        # 检查Youden指数是否有效
        if np.all(np.isnan(Youden_index)) or len(Youden_index) == 0:
            print("警告: Youden指数无效，返回默认阈值0.5")
            return 0.5

        # 找到最佳阈值
        max_youden = np.max(Youden_index)
        best_indices = np.where(Youden_index == max_youden)[0]

        if len(best_indices) == 0:
            print("警告: 无法找到最佳阈值，返回默认阈值0.5")
            return 0.5

        best_threshold = thresholds[best_indices[0]]

        # 确保阈值在合理范围内
        if best_threshold > 1:
            print(f"警告: 阈值{best_threshold}大于1，使用0.5代替")
            best_threshold = 0.5

        print(f"Debug - 找到最佳阈值: {best_threshold}")

        return best_threshold

    except Exception as e:
        print(f"get_best_case_threshold出错: {e}")
        import traceback
        traceback.print_exc()
        return 0.5  # 返回默认阈值


def get_best_threshold(SR, GT):
    """计算最佳阈值（修复版）"""
    try:

        # 转换为numpy数组
        pred_value = SR.data.cpu().numpy() if torch.is_tensor(SR) else np.array(SR)
        true_label = GT.data.cpu().numpy() if torch.is_tensor(GT) else np.array(GT)

        # 检查输入有效性
        if len(pred_value) == 0 or len(true_label) == 0:
            print("警告: 概率或标签为空，返回默认阈值0.5")
            return 0.5

        # 检查标签是否只有一类
        unique_labels = np.unique(true_label)
        if len(unique_labels) < 2:
            print(f"警告: 标签只有{len(unique_labels)}类，返回默认阈值0.5")
            return 0.5

        # 计算ROC曲线
        fpr, tpr, thresholds = metrics.roc_curve(true_label, pred_value)

        # 检查阈值是否有效
        if len(thresholds) == 0:
            print("警告: 无法计算ROC阈值，返回默认阈值0.5")
            return 0.5

        # 排序阈值并取中位数
        thresholds.sort()
        m = int(len(thresholds) / 2)
        best_threshold = thresholds[m]

        # 确保阈值在合理范围内
        if best_threshold > 1:
            print(f"警告: 阈值{best_threshold}大于1，使用0.5代替")
            best_threshold = 0.5

        print(f"Debug - 找到中位数阈值: {best_threshold}")

        return best_threshold

    except Exception as e:
        print(f"get_best_threshold出错: {e}")
        import traceback
        traceback.print_exc()
        return 0.5  # 返回默认阈值

def CalAUC(alllabels, allprobas):
    fpr, tpr, thresholds = roc_curve(alllabels, allprobas, pos_label=1)
    AUC_test = auc(fpr, tpr)
    distances = []
    for i in range(len(thresholds)):
        distances.append(((fpr[i]) ** 2 + (tpr[i] - 1) ** 2))
    a = distances.index(min(distances))
    threshold = thresholds[a]
    return AUC_test, fpr, tpr, threshold

def CalACC(allprobas, alllabels, threshold):
    label_pred = []
    label_true = alllabels
    for i in range(len(alllabels)):
        if allprobas[i] >= threshold:
            label_pred.append(1.0)
        elif allprobas[i] < threshold:
            label_pred.append(0.0)

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(alllabels)):
        if ((label_true[i] == 1) and (label_pred[i] == 1)):
            TP += 1
        elif ((label_true[i] == 0) and (label_pred[i] == 0)):
            TN += 1
        elif ((label_true[i] == 0) and (label_pred[i] == 1)):
            FP += 1
        elif ((label_true[i] == 1) and (label_pred[i] == 0)):
            FN += 1

    specificity = TN / (TN + FP + 1e-12)
    accuracy = (TP + TN) / (TN + TP + FN + FP + 1e-12)
    sensitivity = TP / (TP + FN + 1e-12)
    precision = TP / (TP + FP + 1e-12)
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-12)
    return np.float64(accuracy), np.float64(sensitivity), np.float64(specificity), np.float64(precision), np.float64(F1_score)