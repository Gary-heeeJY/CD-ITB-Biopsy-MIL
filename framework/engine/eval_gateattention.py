import torch
import torch.nn as nn
import torch.nn.functional as F


class Case_predict_gateattention(nn.Module):
    def __init__(self, cfg):
        super(Case_predict_gateattention, self).__init__()

        # === 参数获取 ===
        self.L = cfg.DATA.hidden_dim  # 1024
        self.D = 256  # 注意力内部维度，适当增大以提升容量
        self.K = 1  # 头数
        self.n_classes = cfg.DATA.case_classes

        # 获取 MLP 的膨胀系数，保持和你 ViT 设置一致 (4.0)
        mlp_ratio = cfg.MODEL.Vit.mlp_ratio
        hidden_features = int(self.L * mlp_ratio)  # 1024 * 4 = 4096
        drop_rate = cfg.MODEL.Vit.drop_rate

        # ============================================================
        # 1. Gated Attention (负责从无序切片中提取 Patient 特征)
        # ============================================================
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

        # ============================================================
        # 2. Deep Classifier (负责非线性特征变换，弥补深度不足)
        # ============================================================
        # 这里模拟了 ViT 中的 Feed Forward Network (FFN)
        # 结构: Linear(1024->4096) -> GELU -> Dropout -> Linear(4096->1024) -> ...

        self.deep_classifier = nn.Sequential(
            # 第一层映射：升维，增加特征交互 (1024 -> 4096)
            nn.Linear(self.L * self.K, hidden_features),
            nn.GELU(),  # 使用 GELU，比 ReLU 表现更好，和 Mamba/ViT 保持一致
            nn.Dropout(drop_rate),

            # 第二层映射：降维 (4096 -> 1024)
            nn.Linear(hidden_features, self.L),
            nn.GELU(),
            nn.Dropout(drop_rate),

            # 最终分类层 (1024 -> 2)
            nn.Linear(self.L, self.n_classes)
        )

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, case_feat, slide_logits=None):
        # 1. 维度处理
        if case_feat.dim() == 3:
            case_feat = case_feat.squeeze(0)

        # 2. 计算 Gated Attention Weights
        A_V = self.attention_V(case_feat)
        A_U = self.attention_U(case_feat)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        # 3. 聚合 (Aggregation)
        # M: Patient Level Feature (1, 1024)
        M = torch.mm(A, case_feat)

        # 4. 深层分类 (Deep Classification)
        # 这里不再是简单的 Linear，而是一个 3层的 MLP
        case_logits = self.deep_classifier(M)

        result_dict = {}
        result_dict['case_logit'] = case_logits
        result_dict['case_prob'] = F.softmax(case_logits, dim=1)
        result_dict['attention_weights'] = A

        return result_dict