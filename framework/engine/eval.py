import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ViT_LRP import Block, Mlp


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.attn_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attn_weights = nn.Sequential(nn.Linear(self.D, self.K))

    def forward(self, x):
        A_V = self.attn_V(x)
        A_U = self.attn_U(x)
        attn = A_V.mul(A_U)
        A = self.attn_weights(attn)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        return A


class IndexSelect(nn.Module):
    def __init__(self):
        super(IndexSelect, self).__init__()

    def forward(self, x, dim, indices):
        return torch.index_select(x, dim, indices)


class Case_predict(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.DATA.hidden_dim
        self.n_classes = cfg.DATA.n_classes
        self.case_classes = cfg.DATA.case_classes

        self.attention = Attention_Gated(L=embed_dim, D=256, K=1)
        self.mode = cfg.SOLVER.feat_distillation

        # === 保留 FC 层以兼容旧架构 (ABMIL) ===
        # 即使 MambaMIL 不用它，我们也定义它，防止旧权重加载报错
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(768, self.case_classes)
        )

        self.block = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=cfg.MODEL.Vit.num_heads,
                mlp_ratio=cfg.MODEL.Vit.mlp_ratio,
                qkv_bias=cfg.MODEL.Vit.qkv_bias,
                drop=cfg.MODEL.Vit.drop_rate,
                attn_drop=cfg.MODEL.Vit.attn_drop_rate
            )
            for i in range(cfg.MODEL.Vit.depth)
        ])

        self.case_MLP = Mlp(embed_dim, int(embed_dim * cfg.MODEL.Vit.mlp_ratio), self.case_classes)
        self.case_FC = nn.Linear(embed_dim, self.case_classes)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = IndexSelect()

    def forward(self, case_feat, slide_logits):
        # 1. 蒸馏特征提取
        values, sort_idx = torch.sort(torch.max(slide_logits, dim=1).values, descending=True)
        flat_sort_idx = sort_idx.flatten()
        topk_idx_max = flat_sort_idx[0].item()
        topk_idx_min = flat_sort_idx[-1].item()
        topk_idx = torch.tensor([topk_idx_max, topk_idx_min], dtype=torch.long).to(slide_logits.device)

        if self.mode == 'MaxS':
            feat_dstill = case_feat[topk_idx_max]
        elif self.mode == 'MaxMinS':
            feat_dstill = case_feat.index_select(dim=0, index=topk_idx)
            feat_dstill = torch.sum(feat_dstill, dim=0).unsqueeze(0)
        elif self.mode == 'Mean':
            feat_dstill = torch.mean(case_feat, dim=0).unsqueeze(0)
        else:  # AFS
            tAA = self.attention(case_feat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', case_feat, tAA)
            feat_dstill = torch.sum(tattFeats, dim=0).unsqueeze(0)

        # 2. 维度对齐 (兼容 MambaMIL 和 ABMIL)
        if feat_dstill.ndim == 1:
            feat_dstill_input = feat_dstill.unsqueeze(0).unsqueeze(0)
        elif feat_dstill.ndim == 2:
            feat_dstill_input = feat_dstill.unsqueeze(0)
        else:
            feat_dstill_input = feat_dstill

        if case_feat.ndim == 2:
            case_feat_input = case_feat.unsqueeze(0)
        else:
            case_feat_input = case_feat

        # 3. 拼接与 ViT 处理
        x = torch.cat((feat_dstill_input, case_feat_input), dim=1)

        for blk in self.block:
            x = blk(x)

        x = self.norm(x).squeeze(0)
        case_x = self.pool(x, dim=0, indices=torch.tensor(0, device=x.device))
        case_x = case_x.squeeze()

        if case_x.ndim == 1:
            case_x = case_x.unsqueeze(0)

        case_logit = self.case_MLP(case_x)

        result_dict = {}
        result_dict['case_logit'] = case_logit
        return result_dict