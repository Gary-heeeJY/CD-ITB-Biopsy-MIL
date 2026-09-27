import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODEL_FRAMEWORK_REGISTRY, configurable

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttn(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super(GatedAttn, self).__init__()
        self.L = hidden_dim
        self.D = 128
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(nn.Linear(self.D, self.K),)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

    def forward(self, x):
        x = x.squeeze()
        # x_ = x.clone()
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        attn = A_V.mul(A_U)
        # attn_ = attn.clone()
        A = self.attention_weights(attn) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        sorted_sc, indices = torch.sort(A, descending=True)
        idx = indices.squeeze(0)[:10]
        feat = torch.index_select(x, 0, idx)

        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, x)  # KxL

        Y_prob = self.classifier(M)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        result_dict = {'logits': Y_prob, 'feat': feat}
        return result_dict


@MODEL_FRAMEWORK_REGISTRY.register()
class GatedAttention(nn.Module):
    @configurable
    def __init__(self, hidden_dim, n_classes):
        super(GatedAttention, self).__init__()
        self.L = hidden_dim
        self.D = 128
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(nn.Linear(self.D, self.K),)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

    @classmethod
    def from_config(cls, cfg): #todo
        return {
            "hidden_dim": cfg.DATA.hidden_dim,
            "n_classes": cfg.DATA.n_classes,
        }

    def forward(self, x):
        # x shape: [1, N, 1024] or [N, 1024]
        if x.dim() == 3:
            x = x.squeeze(0)  # [N, 1024]

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        attn = A_V.mul(A_U)
        A = self.attention_weights(attn)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        # === 计算 Attention Score 用于可视化（可选） ===
        A_scores = F.softmax(A, dim=1)  # softmax over N

        # === 计算 Slide 级别特征 (Bag Representation) ===
        # M shape: [K, L] -> [1, 1024]
        M = torch.mm(A_scores, x)

        # === 分类 ===
        Y_prob = self.classifier(M)

        # === 修改处：统一返回接口 ===
        # feat: 必须返回 Slide 级特征 M，以便 Case_predict 将其聚合
        # 原来的 Top-k feat 可能会导致下游维度错误
        result_dict = {
            'logits': Y_prob,
            'feat': M,  # [1, 1024]
            'Y_hat': torch.ge(Y_prob, 0.5).float(),
            'A': A_scores  # 返回注意力权重以便可视化
        }
        return result_dict

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A