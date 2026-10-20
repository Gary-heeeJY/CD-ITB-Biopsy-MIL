import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from .srmamba import SRMamba


def initialize_weights(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            if child.bias is not None:
                child.bias.data.zero_()
        if isinstance(child, nn.LayerNorm):
            nn.init.constant_(child.bias, 0)
            nn.init.constant_(child.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        n_classes=2,
        dropout=0.2,
        act="gelu",
        layer=2,
        rate=10,
        type="SRMamba",
    ):
        super().__init__()
        if type not in {"SRMamba", "Mamba"}:
            raise ValueError(f"Unsupported Mamba type: {type}")

        self.dim = in_dim
        self.type = type
        self.rate = rate

        feature_layers = [nn.Linear(self.dim, self.dim)]
        if act.lower() == "relu":
            feature_layers.append(nn.ReLU())
        elif act.lower() == "gelu":
            feature_layers.append(nn.GELU())
        if dropout > 0:
            feature_layers.append(nn.Dropout(dropout))
        self._fc1 = nn.Sequential(*feature_layers)

        mixer_class = SRMamba if type == "SRMamba" else Mamba
        self.layers = nn.ModuleList()
        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(self.dim),
                    mixer_class(
                        d_model=self.dim,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    ),
                )
            )

        self.norm = nn.LayerNorm(self.dim)
        self.n_classes = n_classes
        self.attention = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Linear(self.dim, self.n_classes)
        self.apply(initialize_weights)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)

        h = self._fc1(x.float())
        for layer in self.layers:
            residual = h
            h = layer[0](h)
            mixer = layer[1]
            if self.type == "SRMamba":
                h = mixer(h, rate=self.rate)
            else:
                h = mixer(h)
            h = h + residual

        h = self.norm(h)
        attention = self.attention(h)
        attention = torch.transpose(attention, 1, 2)
        attention = F.softmax(attention, dim=-1)
        h_slide = torch.bmm(attention, h).squeeze(1)
        logits = self.classifier(h_slide)
        return logits, h_slide, attention
