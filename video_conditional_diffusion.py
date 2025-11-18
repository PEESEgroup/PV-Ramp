import os
import math
import h5py
from pathlib import Path
from random import random
from functools import partial
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import utils as vutils

# ---------------------------------------------
# utils
# ---------------------------------------------


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def divisible_by(n, d):
    return (n % d) == 0


def normalize_to_neg_one_to_one(img):
    # img in [0,1]
    return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    # img in [-1,1]
    return (img + 1) * 0.5


def extract(a, t, x_shape):
    """
    Extract values from a 1-D tensor a at positions t and reshape to x_shape.
    a: [T]
    t: [B]
    returns: [B, 1, 1, 1, 1]
    """
    b = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)


# simple reduction util (equivalent to einops.reduce for mean over all dims except batch)
def reduce_mean_over_nonbatch(x):
    return x.view(x.shape[0], -1).mean(dim=1)


# ---------------------------------------------
# layers
# ---------------------------------------------


class RMSNorm(nn.Module):
    """Simple RMSNorm variant for 3D feature maps (normalizes over spatial+time dims)."""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, C, T, H, W]
        norm_x = x.pow(2).mean(dim=[2, 3, 4], keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return self.weight.view(1, -1, 1, 1, 1) * x


class SinusoidalPosEmb(nn.Module):
    """
    Classic sinusoidal timestep embedding used in DDPM.
    Input: [B] -> [B, dim]
    """

    def __init__(self, dim, theta=10000):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.theta = theta
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.arange(half_dim, dtype=torch.float32)
                * -(math.log(theta) / (half_dim - 1))
            ),
        )

    def forward(self, t):
        # t: [B]
        if t.dtype != torch.float32:
            t = t.float()
        freqs = t[:, None] * self.freqs[None, :]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return emb


class Block3D(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = None
        if exists(time_emb_dim):
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))

        self.block1 = Block3D(dim, dim_out, dropout=dropout)
        self.block2 = Block3D(dim_out, dim_out, dropout=dropout)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)  # [B, 2*dim_out]
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(
                -1
            )  # [B, 2*dim_out, 1, 1, 1]
            scale_shift = time_emb.chunk(
                2, dim=1
            )  # ([B, dim_out,1,1,1], [B, dim_out,1,1,1])

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention3D(nn.Module):
    """Full 3D self-attention over (T,H,W) flattened tokens.
    x: [B, C, T, H, W]
    """

    def __init__(self, dim, heads: int = 4, dim_head: int = 16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)  # [B, inner, T, H, W]

        def reshape_heads(tensor):
            tensor = tensor.view(b, self.heads, self.dim_head, t * h * w)
            return tensor

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        scale = 1.0 / math.sqrt(self.dim_head)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)

        out = (
            out.permute(0, 1, 3, 2)
            .contiguous()
            .view(b, self.heads * self.dim_head, t, h, w)
        )
        out = self.to_out(out)
        return out + x


class TimeAttention3D(nn.Module):
    """Time-only attention: attend along T for each (H,W) independently."""

    def __init__(self, dim, heads: int = 4, dim_head: int = 16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        x_perm = x.permute(0, 3, 4, 2, 1).contiguous()
        x_flat = x_perm.view(b * h * w, t, c)

        qkv = self.to_qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(tensor):
            BHW, T, inner = tensor.shape
            return tensor.view(BHW, T, self.heads, self.dim_head).permute(
                0, 2, 1, 3
            )

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scale = 1.0 / math.sqrt(self.dim_head)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)

        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .view(b * h * w, t, self.heads * self.dim_head)
        )
        out = self.to_out(out)

        out = out.view(b, h, w, t, -1).permute(0, 4, 3, 1, 2).contiguous()
        return out + x


class SpatialAttention3D(nn.Module):
    """Space-only attention: attend over (H,W) for each time slice separately."""

    def __init__(self, dim, heads: int = 4, dim_head: int = 16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, T, H, W, C]
        x_flat = x_perm.view(b * t, h * w, c)

        qkv = self.to_qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(tensor):
            BT, L, inner = tensor.shape
            return tensor.view(BT, L, self.heads, self.dim_head).permute(
                0, 2, 1, 3
            )

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scale = 1.0 / math.sqrt(self.dim_head)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)

        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .view(b * t, h * w, self.heads * self.dim_head)
        )
        out = self.to_out(out)

        out = out.view(b, t, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        return out + x


def Upsample3d(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample3d(dim, dim_out=None):
    return nn.Conv3d(dim, default(dim_out, dim), 3, stride=(1, 2, 2), padding=1)


# ---------------------------------------------
# 3D UNet (video conditional)
# ---------------------------------------------


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        init_dim=None,
        attn_type="time",
        input_cond_channels=3,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        learned_variance=False,
        dropout=0.0,
    ):
        super().__init__()

        self.channels = channels
        self.attn_type = attn_type
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(input_channels, init_dim, 7, padding=3)
        self.init_conv_cond = nn.Conv3d(input_cond_channels, init_dim, 7, padding=3)

        # time embedding for diffusion steps
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        resnet_block = partial(ResnetBlock3D, time_emb_dim=time_dim, dropout=dropout)

        self.downs = nn.ModuleList([])
        self.downs_cond = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # choose attention type per resolution
            if self.attn_type == "alternate":
                CurAttn = TimeAttention3D if (ind % 2 == 0) else SpatialAttention3D
            elif self.attn_type == "time":
                CurAttn = TimeAttention3D
            elif self.attn_type == "space":
                CurAttn = SpatialAttention3D
            elif self.attn_type == "none":
                CurAttn = lambda dim: nn.Identity()
            else:
                CurAttn = Attention3D

            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        CurAttn(dim_in),
                        Downsample3d(dim_in, dim_out)
                        if not is_last
                        else nn.Conv3d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

            self.downs_cond.append(
                nn.ModuleList(
                    [
                        ResnetBlock3D(dim_in, dim_in, time_emb_dim=None),
                        ResnetBlock3D(dim_in, dim_in, time_emb_dim=None),
                        CurAttn(dim_in),
                        Downsample3d(dim_in, dim_out)
                        if not is_last
                        else nn.Conv3d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim * 2, mid_dim)
        self.mid_attn = (
            TimeAttention3D(mid_dim) if attn_type != "none" else nn.Identity()
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            # mirror encoder attention pattern for decoder
            enc_ind = len(in_out) - 1 - ind
            if self.attn_type == "alternate":
                CurAttnUp = TimeAttention3D if (enc_ind % 2 == 0) else SpatialAttention3D
            elif self.attn_type == "time":
                CurAttnUp = TimeAttention3D
            elif self.attn_type == "space":
                CurAttnUp = SpatialAttention3D
            elif self.attn_type == "none":
                CurAttnUp = lambda dim: nn.Identity()
            else:
                CurAttnUp = Attention3D

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out + dim_in * 2, dim_out),
                        resnet_block(dim_out + dim_in * 2, dim_out),
                        CurAttnUp(dim_out),
                        Upsample3d(dim_out, dim_in)
                        if not is_last
                        else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 3, init_dim)
        self.final_conv = nn.Conv3d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, cond_input, time, x_self_cond=None):
        """
        x:          [B, C, T_real, H, W]  (noised future video, e.g. 16 frames)
        cond_input: [B, C, T_cond, H, W]  (4 past + 16 blurry future)
        time:       [B] (diffusion step)
        """
        target_dtype = self.init_conv.weight.dtype
        x = x.to(dtype=target_dtype)
        cond_input = cond_input.to(dtype=target_dtype)

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        x_cond = self.init_conv_cond(cond_input)

        if x_cond.shape[2:] != x.shape[2:]:
            x_cond = F.interpolate(
                x_cond,
                size=x.shape[2:],  # (T_real, H, W)
                mode="trilinear",
                align_corners=False,
            )

        r_cond = x_cond.clone()

        t_emb = self.time_mlp(time)  # [B, time_dim]

        hs = []
        for (block1, block2, attn, downsample) in self.downs:
            x = block1(x, t_emb)
            hs.append(x)
            x = block2(x, t_emb)
            x = attn(x) + x
            hs.append(x)
            x = downsample(x)

        hs_cond = []
        for (block1, block2, attn, downsample) in self.downs_cond:
            x_cond = block1(x_cond, None)
            hs_cond.append(x_cond)
            x_cond = block2(x_cond, None)
            x_cond = attn(x_cond) + x_cond
            hs_cond.append(x_cond)
            x_cond = downsample(x_cond)

        # bottleneck
        x = torch.cat((x, x_cond), dim=1)
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t_emb)

        for (block1, block2, attn, upsample) in self.ups:
            x = torch.cat((x, hs.pop(), hs_cond.pop()), dim=1)
            x = block1(x, t_emb)
            x = torch.cat((x, hs.pop(), hs_cond.pop()), dim=1)
            x = block2(x, t_emb)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r, r_cond), dim=1)
        x = self.final_res_block(x, t_emb)
        return self.final_conv(x)


# ---------------------------------------------
# diffusion helpers
# ---------------------------------------------


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@dataclass
class ModelPrediction:
    pred_noise: torch.Tensor
    pred_x_start: torch.Tensor


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_noise",
        offset_noise_strength=0.0,
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    ):
        super().__init__()

        self.model = model
        self.channels = model.channels

        self.objective = objective
        assert objective in {"pred_noise", "pred_x0", "pred_v"}

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        )
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

        # p2 loss reweighting
        self.register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

        self.offset_noise_strength = offset_noise_strength

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def model_predictions(self, x, cond_input, t, x_self_cond=None, clip_x_start=False):
        model_out = self.model(x, cond_input, t, x_self_cond)

        if self.objective == "pred_noise":
            pred_noise = model_out
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        elif self.objective == "pred_x0":
            x_start = model_out
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_v":
            v = model_out
            x_start = (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * v
            )
            pred_noise = (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * v
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
            )
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if clip_x_start:
            x_start = x_start.clamp(-1.0, 1.0)

        return ModelPrediction(pred_noise=pred_noise, pred_x_start=x_start)

    def p_mean_variance(self, x, cond_input, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(
            x, cond_input, t, x_self_cond, clip_x_start=clip_denoised
        )
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, _, model_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, model_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, cond_input, t_idx, x_self_cond=None):
        b = x.shape[0]
        batched_times = torch.full((b,), t_idx, device=x.device, dtype=torch.long)
        model_mean, model_log_variance, x_start = self.p_mean_variance(
            x, cond_input, batched_times, x_self_cond=x_self_cond
        )
        if t_idx == 0:
            return model_mean, x_start
        noise = torch.randn_like(x)
        return model_mean + (0.5 * model_log_variance).exp() * noise, x_start

    @torch.no_grad()
    def p_sample_loop(self, cond_input, shape, return_all_timesteps=False):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        imgs = [img]

        x_self_cond = None

        for i in reversed(range(self.num_timesteps)):
            img, x_start = self.p_sample(
                img, cond_input, i, x_self_cond=x_self_cond
            )
            x_self_cond = x_start
            if return_all_timesteps:
                imgs.append(img)

        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        return img

    @torch.no_grad()
    def ddim_sample(self, cond_input, shape, return_all_timesteps=False):
        """Deterministic DDIM sampling over video volume."""
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = [img]

        # time steps for DDIM (subsampled or full)
        times = torch.linspace(
            0, self.num_timesteps - 1, self.sampling_timesteps, device=device
        ).long()
        times = list(reversed(times.tolist()))

        for i, t_idx in enumerate(times):
            t = torch.full((b,), t_idx, device=device, dtype=torch.long)

            preds = self.model_predictions(img, cond_input, t, clip_x_start=True)
            x0, e_t = preds.pred_x_start, preds.pred_noise

            if i == len(times) - 1:
                img = x0
                if return_all_timesteps:
                    imgs.append(img)
                break

            prev_t_idx = times[i + 1]
            t_prev = torch.full((b,), prev_t_idx, device=device, dtype=torch.long)

            alpha_t = extract(self.alphas_cumprod, t, img.shape)
            alpha_prev = extract(self.alphas_cumprod, t_prev, img.shape)

            # DDIM update with eta = 0 (deterministic)
            sigma = 0.0
            sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)
            dir_xt = sqrt_one_minus_alpha_prev * e_t
            img = torch.sqrt(alpha_prev) * x0 + dir_xt

            if sigma > 0:
                img = img + sigma * torch.randn_like(img)

            if return_all_timesteps:
                imgs.append(img)

        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        return img

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, cond_input, x_start, t, noise=None, offset_noise_strength=None):
        """
        x_start: [B, C, T_real, H, W]
        cond_input: [B, C, T_cond, H, W]
        """
        cond_input = cond_input.to(dtype=torch.float32)
        x_start = x_start.to(dtype=torch.float32)

        if noise is None:
            noise = torch.randn_like(x_start)

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=x_start.device)
            # broadcast to all spatial dims
            noise = noise + offset_noise_strength * offset_noise[
                :, :, None, None, None
            ]

        x_noised = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_self_cond = None
        model_out = self.model(x_noised, cond_input, t, x_self_cond=x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * x_start
            )
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce_mean_over_nonbatch(loss)
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, cond, real, *args, **kwargs):
        """
        cond: [B, C, T_cond, H, W]
        real: [B, C, T_real, H, W]
        """
        b = real.shape[0]
        device = real.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(cond_input=cond, x_start=real, t=t, *args, **kwargs)

    @torch.no_grad()
    def sample(self, cond_input, target_T=None, batch_size=None, return_all_timesteps=False):
        """
        Sample future video given conditions using DDPM or DDIM.

        cond_input: [B, C, T_cond, H, W]
        target_T:   number of frames to generate (e.g. 16). If None, use T_cond.
        """
        if batch_size is None:
            batch_size = cond_input.shape[0]
        _, _, T_cond, H, W = cond_input.shape
        if target_T is None:
            target_T = T_cond
        shape = (batch_size, self.channels, target_T, H, W)
        sample_fn = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop
        return sample_fn(cond_input, shape, return_all_timesteps=return_all_timesteps)


# ---------------------------------------------
# dataset
# ---------------------------------------------


class BlurryVideoDataset(Dataset):
    """
    real_vids: [N, C, T_real, H, W]
    cond_vids: [N, C, T_cond, H, W]  (T_cond 可以 != T_real)
    """

    def __init__(self, cond_vids, real_vids):
        super().__init__()
        assert cond_vids.shape[0] == real_vids.shape[0]
        assert cond_vids.shape[1] == real_vids.shape[1]
        assert cond_vids.shape[-2:] == real_vids.shape[-2:]
        self.cond_vids = cond_vids
        self.real_vids = real_vids

    def __len__(self):
        return self.real_vids.shape[0]

    def __getitem__(self, idx):
        cond = self.cond_vids[idx].float()
        real = self.real_vids[idx].float()
        return cond, real


def load_video_dataset(hdf5_path):
    """
    Assumes HDF5 file contains:
      - real_vids: (N, T_real=16, H, W, 3)
      - pred_vids: (N, T_pred=16, H, W, 3)  (PhyDNet blurry future)
      - prev_vids: (N, T_prev=4,  H, W, 3)  (history frames)
    We build:
      - cond_vids_time = concat(prev_vids, pred_vids, dim=time) -> (N, 20, H, W, 3)
      - real_vids       -> (N, 16, H, W, 3)
    And convert to torch:
      cond_vids: [N, 3, 20, H, W]
      real_vids: [N, 3, 16, H, W]
    """
    with h5py.File(hdf5_path, "r") as f:
        real_vids = f["real_vids"][...]  # (N, T_real, H, W, 3)
        pred_vids = f["pred_vids"][...]  # (N, T_pred, H, W, 3)
        prev_vids = f["prev_vids"][...]  # (N, T_prev, H, W, 3)

    real_vids = np.asarray(real_vids, dtype=np.float32) / 255.0
    pred_vids = np.asarray(pred_vids, dtype=np.float32) / 255.0
    prev_vids = np.asarray(prev_vids, dtype=np.float32) / 255.0

    real_vids = (
        torch.from_numpy(real_vids)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
        .float()
    )
    pred_vids = (
        torch.from_numpy(pred_vids)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
        .float()
    )
    prev_vids = (
        torch.from_numpy(prev_vids)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
        .float()
    )

    real_vids = normalize_to_neg_one_to_one(real_vids)
    pred_vids = normalize_to_neg_one_to_one(pred_vids)
    prev_vids = normalize_to_neg_one_to_one(prev_vids)

    cond_vids = torch.cat([prev_vids, pred_vids], dim=2)  # (N,3,T_cond,H,W)

    return cond_vids, real_vids


# ---------------------------------------------
# trainer
# ---------------------------------------------


class Trainer:
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        *,
        dataset_path,
        train_batch_size=8,
        train_num_steps=100000,
        learning_rate=1e-4,
        save_and_sample_every=1000,
        results_folder="./results_video",
        num_workers=4,
        device=None,
    ):
        super().__init__()

        self.model = diffusion_model
        self.dataset_path = dataset_path
        self.train_batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.save_and_sample_every = save_and_sample_every

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        cond_vids, real_vids = load_video_dataset(dataset_path)
        dataset = BlurryVideoDataset(cond_vids, real_vids)
        self.dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.step = 0

    def save_checkpoint(self, milestone):
        data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }
        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load_checkpoint(self, path):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.step = data.get("step", 0)

    def train(self):
        self.model.train()
        while self.step < self.train_num_steps:
            for cond_vids, real_vids in self.dl:
                cond_vids = cond_vids.to(self.device, dtype=torch.float32)
                real_vids = real_vids.to(self.device, dtype=torch.float32)

                loss = self.model(cond=cond_vids, real=real_vids)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.step % 20 == 0:
                    print(
                        f"step {self.step} / {self.train_num_steps}, loss: {loss.item():.4f}"
                    )

                if self.step > 0 and self.step % self.save_and_sample_every == 0:
                    self.save_checkpoint(self.step)
                    self.sample_and_save(self.step, cond_vids, real_vids)

                self.step += 1
                if self.step >= self.train_num_steps:
                    break

    @torch.no_grad()
    def sample_and_save(self, milestone, cond_vids, real_vids):
        self.model.eval()
        cond_vids = cond_vids[:4]  # take a small batch to visualize
        real_vids = real_vids[:4]

        target_T = real_vids.shape[2]
        samples = self.model.sample(cond_vids, target_T=target_T)  # [B, C, T_real, H, W]

        B, C, T_cond, H, W = cond_vids.shape
        _, C, T_real, _, _ = real_vids.shape

        T_prev = max(T_cond - T_real, 0)

        if T_prev > 0:
            prev_vids = cond_vids[:, :, :T_prev, :, :]  # [B,3,T_prev,H,W]
            pred_vids = cond_vids[:, :, T_prev:, :, :]  # [B,3,T_real,H,W]
        else:
            prev_vids = cond_vids[:, :, :1, :, :] 
            pred_vids = cond_vids

        t_prev_idx = min(T_prev - 1, prev_vids.shape[2] - 1) if T_prev > 0 else 0
        t0 = 0

        prev_imgs = prev_vids[:, :, t_prev_idx, :, :]
        pred_imgs = pred_vids[:, :, t0, :, :]
        sample_imgs = samples[:, :, t0, :, :]
        real_imgs = real_vids[:, :, t0, :, :]

        # concatenate along "tile" axis: [prev, pred, sample, real]
        # shape: [B, 4, C, H, W] -> [C, B*H, 4*W]
        merge = torch.stack([prev_imgs, pred_imgs, sample_imgs, real_imgs], dim=1)
        merge = merge.permute(2, 0, 3, 1, 4)  # [C, B, H, 4, W]
        merge = merge.reshape(C, B * H, 4 * W)

        merge = unnormalize_to_zero_to_one(merge).clamp_(0.0, 1.0)
        vutils.save_image(merge, str(self.results_folder / f"sample-{milestone}.png"))
        self.model.train()


# ---------------------------------------------
# main
# ---------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Video Conditional Diffusion"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to HDF5 file (N,T,H,W,3)"
    )
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--results", type=str, default="./results_video")
    args = parser.parse_args()

    unet3d = Unet3D(
        dim=64,
        init_dim=None,
        attn_type="time",  
        input_cond_channels=3,   
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        learned_variance=False,
        dropout=0.0,
    )

    diffusion = GaussianDiffusion(
        unet3d,
        timesteps=1000,
        sampling_timesteps=1000,
        objective="pred_noise",
        offset_noise_strength=0.0,
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    )

    trainer = Trainer(
        diffusion,
        dataset_path=args.data,
        train_batch_size=args.batch_size,
        train_num_steps=args.steps,
        learning_rate=args.lr,
        save_and_sample_every=args.save_every,
        results_folder=args.results,
    )

    trainer.train()
