"""Case-level aggregation variants reported by the paper.

A and D are the two settings used before this public release was organized.
B and C isolate the attention and Transformer components respectively so the
four-way ablation uses the same slide encoder, classifier, and training loop.
"""

import torch
import torch.nn as nn

from engine.eval import Attention_Gated
from models.ViT_LRP import Block, Mlp


class Case_predict_variant(nn.Module):
    def __init__(self, cfg, use_afs=False, use_transformer=False, use_cls_token=False):
        super().__init__()
        embed_dim = cfg.DATA.hidden_dim
        self.case_classes = cfg.DATA.case_classes
        self.use_afs = use_afs
        self.use_transformer = use_transformer
        self.use_cls_token = use_cls_token
        self.mean_pool = not use_afs and not use_transformer and not use_cls_token
        self.attention = Attention_Gated(L=embed_dim, D=256, K=1) if use_afs else None

        depth = cfg.MODEL.Vit.depth if use_transformer else 0
        if depth:
            self.block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=cfg.MODEL.Vit.num_heads,
                        mlp_ratio=cfg.MODEL.Vit.mlp_ratio,
                        qkv_bias=cfg.MODEL.Vit.qkv_bias,
                        drop=cfg.MODEL.Vit.drop_rate,
                        attn_drop=cfg.MODEL.Vit.attn_drop_rate,
                    )
                    for _ in range(depth)
                ]
            )
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.block = None
            self.norm = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None
        self.case_MLP = Mlp(embed_dim, int(embed_dim * cfg.MODEL.Vit.mlp_ratio), self.case_classes)

    @staticmethod
    def _to_ndim2(case_feat):
        return case_feat.squeeze(0) if case_feat.ndim == 3 else case_feat

    def forward(self, case_feat, slide_logits):
        case_feat = self._to_ndim2(case_feat)
        if self.mean_pool:
            return {"case_logit": self.case_MLP(torch.mean(case_feat, dim=0, keepdim=True))}

        if self.use_afs:
            attention = self.attention(case_feat).squeeze(0)
            token = torch.sum(case_feat * attention.unsqueeze(1), dim=0, keepdim=True)
        elif self.use_cls_token:
            token = self.cls_token
        else:
            token = torch.mean(case_feat, dim=0, keepdim=True).unsqueeze(0)

        if not self.use_transformer:
            return {"case_logit": self.case_MLP(token.squeeze(0) if token.ndim == 3 else token)}

        if token.ndim == 2:
            token = token.unsqueeze(0)
        x = torch.cat((token, case_feat.unsqueeze(0)), dim=1)
        for block in self.block:
            x = block(x)
        x = self.norm(x).squeeze(0)
        return {"case_logit": self.case_MLP(x[0].unsqueeze(0))}


VARIANT_FLAGS = {
    # Mean pooling: no attention token and no Transformer.
    "A": {"use_afs": False, "use_transformer": False, "use_cls_token": False},
    # Attention-only: gated attention produces the case token, then MLP.
    "B": {"use_afs": True, "use_transformer": False, "use_cls_token": False},
    # Transformer-only: learned case token attends to slide tokens.
    "C": {"use_afs": False, "use_transformer": True, "use_cls_token": True},
    # Proposed model: attention-derived token plus Transformer fusion.
    "D": {"use_afs": True, "use_transformer": True, "use_cls_token": False},
}


def build_case_predict(variant, cfg):
    variant = str(variant).upper()
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"Unknown case aggregation variant: {variant}")
    return Case_predict_variant(cfg, **VARIANT_FLAGS[variant])


class Case_predict_MeanPool(Case_predict_variant):
    def __init__(self, cfg):
        super().__init__(cfg, **VARIANT_FLAGS["A"])


class Case_predict_AFSOnly(Case_predict_variant):
    def __init__(self, cfg):
        super().__init__(cfg, **VARIANT_FLAGS["B"])


class Case_predict_TransOnly(Case_predict_variant):
    def __init__(self, cfg):
        super().__init__(cfg, **VARIANT_FLAGS["C"])


class Case_predict_Full(Case_predict_variant):
    def __init__(self, cfg):
        super().__init__(cfg, **VARIANT_FLAGS["D"])
