import torch
import torch.nn as nn
import torch.nn.functional as F

# 强制使用系统安装的 Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("请确保已安装 mamba_ssm: pip install mamba_ssm")


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, in_dim=1024, n_classes=2, dropout=0.2, act='gelu', layer=2, rate=10, type="SRMamba"):
        super(MambaMIL, self).__init__()

        # 保存输入维度，不再硬编码 512
        self.dim = in_dim

        # 1. 特征变换层
        self._fc1 = [nn.Linear(self.dim, self.dim)]  # 保持维度不变
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout > 0:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)

        # 2. Mamba 核心层堆叠
        self.layers = nn.ModuleList()
        self.type = type
        self.rate = rate

        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(self.dim),
                    Mamba(
                        d_model=self.dim,  # 动态使用输入维度 (1024)
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    )
                )
            )

        self.norm = nn.LayerNorm(self.dim)
        self.n_classes = n_classes

        # 3. Attention Pooling
        self.attention = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 4. 分类头
        self.classifier = nn.Linear(self.dim, self.n_classes)
        self.apply(initialize_weights)

    def forward(self, x):
        # x shape: [Batch, N, Dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        h = x.float()
        h = self._fc1(h)

        # 残差连接逻辑
        if self.type == "SRMamba" or self.type == "Mamba":
            for layer in self.layers:
                h_ = h
                output = layer(h)
                h = output + h_
        else:
            for layer in self.layers:
                h = layer(h)

        h = self.norm(h)

        # Attention Aggregation
        A = self.attention(h)  # [B, N, 1]
        A = torch.transpose(A, 1, 2)  # [B, 1, N]
        A = F.softmax(A, dim=-1)  # [B, 1, N]

        h_slide = torch.bmm(A, h)  # [B, 1, Dim]
        h_slide = h_slide.squeeze(1)  # [B, Dim] (1024)

        logits = self.classifier(h_slide)

        # 【修改】返回 Logits, Slide特征, 以及 Attention权重(A)
        # 兼容性处理：如果是在推理模式且不需要A，可以不接；但为了画图我们返回它
        # A 的形状是 [1, 1, N]，画图时需要 squeeze 成 [N]
        return logits, h_slide, A