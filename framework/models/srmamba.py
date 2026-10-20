import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class TransposeTokenReEmbedding:
    @staticmethod
    def transpose_normal_padding(x, rate):
        if rate < 1:
            raise ValueError("rate must be a positive integer")
        x = rearrange(x, "b c l -> b l c")
        _, length, _ = x.shape
        value = length // rate
        if length % rate != 0:
            padding_length = (value + 1) * rate - length
            x = F.pad(x, (0, 0, 0, padding_length))
        x = rearrange(x, "b (k w) d -> b (w k) d", w=rate)
        return rearrange(x, "b l c -> b c l")

    @staticmethod
    def transpose_remove_padding(x, rate, length):
        x = rearrange(x, "b c l -> b l c")
        x = rearrange(x, "b (w k) d -> b (k w) d", w=rate)
        x = x[:, :length, :]
        return rearrange(x, "b l c -> b c l")


class SRMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = self._build_conv(d_conv, conv_bias, factory_kwargs)
        self.conv1d_b = self._build_conv(d_conv, conv_bias, factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.x_proj_b = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self._initialize_dt(self.dt_proj, dt_min, dt_max, dt_init, dt_scale, dt_init_floor)
        self._initialize_dt(self.dt_proj_b, dt_min, dt_max, dt_init, dt_scale, dt_init_floor)

        state = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(state))
        self.A_log._no_weight_decay = True
        self.A_b_log = nn.Parameter(torch.log(state.clone()))
        self.A_b_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_b._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def _build_conv(self, d_conv, conv_bias, factory_kwargs):
        return nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )

    def _initialize_dt(self, projection, dt_min, dt_max, dt_init, dt_scale, dt_init_floor):
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(projection.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(projection.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(self.d_inner, device=projection.weight.device)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            projection.bias.copy_(inverse_softplus)
        projection.bias._no_reinit = True

    def _causal_convolution(self, x, convolution):
        length = x.shape[-1]
        if causal_conv1d_fn is not None:
            return causal_conv1d_fn(
                x=x,
                weight=rearrange(convolution.weight, "d 1 w -> d w"),
                bias=convolution.bias,
                activation="silu",
            )
        x = F.conv1d(
            x,
            convolution.weight,
            convolution.bias,
            padding=self.d_conv - 1,
            groups=self.d_inner,
        )
        return F.silu(x[..., :length])

    def _scan_branch(self, xz, convolution, x_projection, dt_projection, a_log, skip):
        x, z = xz.chunk(2, dim=1)
        x = self._causal_convolution(x, convolution)
        x_dbl = x_projection(rearrange(x, "b d l -> (b l) d"))
        delta, b_state, c_state = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = dt_projection(delta)
        delta = rearrange(delta, "(b l) d -> b d l", l=x.shape[-1])
        b_state = rearrange(
            b_state,
            "(b l) dstate -> b dstate l",
            l=x.shape[-1],
        ).contiguous()
        c_state = rearrange(
            c_state,
            "(b l) dstate -> b dstate l",
            l=x.shape[-1],
        ).contiguous()
        return selective_scan_fn(
            x,
            delta,
            -torch.exp(a_log.float()),
            b_state,
            c_state,
            skip.float(),
            z=z,
            delta_bias=dt_projection.bias.float(),
            delta_softplus=True,
        )

    def forward(self, hidden_states, inference_params=None, rate=10):
        if inference_params is not None:
            raise NotImplementedError("SRMamba inference cache is not implemented")
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, dimension)")
        if not isinstance(rate, int) or rate < 1:
            raise ValueError("rate must be a positive integer")

        xz = rearrange(self.in_proj(hidden_states), "b l d -> b d l")
        sequence_length = xz.shape[-1]
        out_forward = self._scan_branch(
            xz,
            self.conv1d,
            self.x_proj,
            self.dt_proj,
            self.A_log,
            self.D,
        )
        reordered_xz = TransposeTokenReEmbedding.transpose_normal_padding(xz, rate)
        out_reordered = self._scan_branch(
            reordered_xz,
            self.conv1d_b,
            self.x_proj_b,
            self.dt_proj_b,
            self.A_b_log,
            self.D_b,
        )
        out_reordered = TransposeTokenReEmbedding.transpose_remove_padding(
            out_reordered,
            rate,
            sequence_length,
        )
        output = out_forward + out_reordered
        return self.out_proj(rearrange(output, "b d l -> b l d"))
