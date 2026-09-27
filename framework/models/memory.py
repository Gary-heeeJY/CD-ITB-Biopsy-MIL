import torch
import torch.nn as nn
import numpy as np
from .build import MODEL_FRAMEWORK_REGISTRY, configurable

@MODEL_FRAMEWORK_REGISTRY.register()
class Memory(nn.Module):
    @configurable
    def __init__(self, mu_queue_len=1, n_classes=3, out_feat_num=1024):
        super().__init__()
        self.n_mu = mu_queue_len
        self.n_class = n_classes
        self.out_dim = out_feat_num

        self.register_buffer("cls_mu_queue", torch.zeros(self.n_class, self.n_mu, self.out_dim))
        self.register_buffer("cls_sc_queue", torch.zeros(self.n_class, self.n_mu))

    @classmethod
    def from_config(cls, cfg): #todo
        return {
            "n_classes": cfg.DATA.n_classes,
            "mu_queue_len": cfg.DATA.mu_queue_len,
            "out_feat_num": cfg.MODEL.out_feat_num
        }

    @torch.no_grad()
    def _update_queue(self, inp_mu, inp_sc, cls_idx):
        for idx in cls_idx:
            self._sort_permutation(inp_mu, inp_sc, idx)

    @torch.no_grad()
    def _sort_permutation(self, inp_mu, inp_sc, idx):
        concat_sc = torch.cat([self.cls_sc_queue[idx, ...], inp_sc[..., idx]], 0)
        concat_mu = torch.cat([self.cls_mu_queue[idx, ...], inp_mu], 0)
        sorted_sc, indices = torch.sort(concat_sc, descending=True)
        sorted_mu = torch.index_select(concat_mu, 0, indices[:self.n_mu])
        self.cls_mu_queue[idx, ...] = sorted_mu
        self.cls_sc_queue[idx, ...] = sorted_sc[:self.n_mu]

    @torch.no_grad()
    def _init_queue(self, mu_queue, sc_queue, lbl_queue):
        for mu, sc, lbl in zip(mu_queue, sc_queue, lbl_queue):
            idxs = np.where(lbl==1)[0].tolist()
            self._update_queue(mu, sc, idxs)

    @torch.no_grad()
    def _return_queue(self, cls_idx):
        mus = []
        for idx in cls_idx:
            mus.append(self.cls_mu_queue[idx][None, ...])
        mus = torch.cat(mus, 1)
        return mus