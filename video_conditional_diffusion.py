import argparse
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from diff_modules import *

# ---------------------------------------------
# 3D UNet (video conditional)
# ---------------------------------------------

class PassthroughCrossAttn(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, context=None):
        return x

class NullCrossAttn(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, context=None):
        return torch.zeros_like(x)

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        init_dim=None,
        attn_type=None,
        cross_attn=None,
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
        self.init_dim = init_dim
        self.init_conv = nn.Conv3d(input_channels, init_dim, 7, padding=3)
        self.init_conv_cond = nn.Conv3d(input_cond_channels, init_dim, 7, padding=3)

        # time emb
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # multi-res
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        self.dims = dims
        num_resolutions = len(dims) - 1
        in_out = list(zip(dims[:-1], dims[1:]))
        
        valid_kinds = {"time","space","none","full"}
        if isinstance(self.attn_type, (list, tuple)):
            if len(self.attn_type) != num_resolutions:
                raise ValueError(f"attn_type length must be {num_resolutions}, got {len(self.attn_type)}")
            for v in self.attn_type:
                if v not in valid_kinds:
                    raise ValueError(f"Invalid attn_type value: {v}. Choose from {valid_kinds}.")
            attn_kinds = list(self.attn_type)
        elif isinstance(self.attn_type, str):
            if self.attn_type not in valid_kinds:
                raise ValueError(f"Invalid attn_type value: {self.attn_type}. Choose from {valid_kinds}.")
            attn_kinds = [self.attn_type] * num_resolutions
        elif self.attn_type is None:
            attn_kinds = ["time"] * num_resolutions
        else:
            raise TypeError("attn_type must be list/tuple[str], str, or None")
        
        
        # --- normalize per-layer cross-attention mask ---
        if cross_attn is None:
            cross_mask = [False] * num_resolutions
        elif isinstance(cross_attn, (list, tuple)):
            if len(cross_attn) != num_resolutions:
                raise ValueError(f"cross_attn length must be {num_resolutions}, got {len(cross_attn)}")
            cross_mask = [bool(v) for v in cross_attn]
        else:
            raise TypeError("cross_attn must be list/tuple[bool] or None")
        
        resnet_block = partial(ResnetBlock3D, time_emb_dim=time_dim, dropout=dropout)

        def get_attn(ind, dim_in):
            kind = attn_kinds[ind]
            if kind == "time":
                CurAttn = LinearTimeAttention3D
            elif kind == "space":
                CurAttn = LinearSpatialAttention3D
            elif kind == "none":
                return nn.Identity()
            elif kind == "full":
                CurAttn = LinearAttention3D
            else:
                raise ValueError(f"Unknown attn kind: {kind}")
            return CurAttn(dim_in)

        # down
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            block1 = resnet_block(dim_in, dim_out)
            block2 = resnet_block(dim_out, dim_out)
            attn = get_attn(ind, dim_out)
            downsample = Downsample3d(dim_out, dim_out) if not is_last else nn.Identity()
            self.downs.append(nn.ModuleList([block1, block2, attn, downsample]))

        # mid
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = LinearAttention3D(mid_dim, heads=4, dim_head=16)
        self.mid_cross_attn = LinearCrossAttention3D(mid_dim, mid_dim) if (cross_mask[-1]) else PassthroughCrossAttn()
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        # up 
        self.ups = nn.ModuleList([])
        self.cross_attn_up_tex = nn.ModuleList([])
        self.cross_attn_up_str = nn.ModuleList([])
        self.gate_tex = nn.Parameter(torch.zeros(num_resolutions))
        self.gate_str = nn.Parameter(torch.zeros(num_resolutions))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            enc_ind = num_resolutions - 1 - ind
            CurAttnUp = get_attn(enc_ind, dim_in)

            block1 = resnet_block(dim_out + dim_out, dim_in)
            block2 = resnet_block(dim_in, dim_in)
            is_last = ind == (len(in_out) - 1)
            upsample = Upsample3d(dim_in, dim_in) if not is_last else nn.Identity()
            self.ups.append(nn.ModuleList([block1, block2, CurAttnUp, upsample]))

            use_cross = cross_mask[enc_ind]
            cond_dim_at_scale = dims[enc_ind + 1]
            if use_cross:
                self.cross_attn_up_tex.append(LinearCrossAttention3D(dim_in, cond_dim_at_scale))
                self.cross_attn_up_str.append(LinearCrossAttention3D(dim_in, cond_dim_at_scale))
            else:
                self.cross_attn_up_tex.append(NullCrossAttn())
                self.cross_attn_up_str.append(NullCrossAttn())

        # dual cond encoders
        self.past_encoder = CondEncoder3D(dims, dropout=dropout)
        self.blur_encoder = CondEncoder3D(dims, dropout=dropout)

        # fused for down/mid
        self.cond_fuse = nn.ModuleList(
            [nn.Conv3d(dims[i + 1] * 2, dims[i + 1], 1) for i in range(num_resolutions)]
        )

        # down cross-attn
        self.cross_attn_down = nn.ModuleList([])
        for i in range(num_resolutions):
            use_cross = cross_mask[i]
            if use_cross:
                self.cross_attn_down.append(LinearCrossAttention3D(dims[i + 1], dims[i + 1]))
            else:
                self.cross_attn_down.append(PassthroughCrossAttn())

        # head
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = resnet_block(init_dim * 3, init_dim)
        self.final_conv = nn.Conv3d(init_dim, self.out_dim, 1)
        
    @property
    def downsample_factor(self):
        return 2 ** (len(self.dims) - 2)

    @staticmethod
    def _resize_to_context(context: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if context.shape[2:] == ref.shape[2:]:
            return context
        return F.interpolate(context, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def _prepare_cond_features(self, cond_input):
        B, C, T_cond, H, W = cond_input.shape
        assert T_cond >= 4, "cond_input time dimension must be >= 4 (4 past + future)"
        T_past = 4
        past = cond_input[:, :, :T_past, :, :]
        blur = cond_input[:, :, T_past:, :, :]

        past_feat0 = self.init_conv_cond(past)
        blur_feat0 = self.init_conv_cond(blur)

        F_past = self.past_encoder(past_feat0)
        F_blur = self.blur_encoder(blur_feat0)

        F_cond = [fuse(torch.cat([
                    (fp if fp.shape[2:] == fb.shape[2:] else
                        F.interpolate(fp, size=fb.shape[2:], mode="trilinear", align_corners=False)),
                    fb
                ], dim=1))
                  for fuse, fp, fb in zip(self.cond_fuse, F_past, F_blur)]

        return F_cond, F_past, F_blur

    def forward(self, x, cond_input, time, x_self_cond=None):
        target_dtype = self.init_conv.weight.dtype
        x = x.to(dtype=target_dtype)
        cond_input = cond_input.to(dtype=target_dtype)

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        cond_full = self.init_conv_cond(cond_input)
        F_cond, F_past, F_blur = self._prepare_cond_features(cond_input)

        t_emb = self.time_mlp(time)

        # down
        hs = []
        cur = x
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            cur = block1(cur, t_emb)
            cur = block2(cur, t_emb)
            cur = attn(cur)
            cond_i = self._resize_to_context(F_cond[i], cur)
            cur = self.cross_attn_down[i](cur, cond_i)
            hs.append(cur)
            cur = downsample(cur)

        # mid
        cur = self.mid_block1(cur, t_emb)
        cur = self.mid_attn(cur)
        cond_mid = self._resize_to_context(F_cond[-1], cur)
        cur = self.mid_cross_attn(cur, cond_mid)
        cur = self.mid_block2(cur, t_emb)

        # up
        F_past_up = list(reversed(F_past))
        F_blur_up = list(reversed(F_blur))
        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            skip = hs.pop()
            cur = torch.cat([cur, skip], dim=1)
            cur = block1(cur, t_emb)
            cur = block2(cur, t_emb)
            cur = attn(cur)

            g_tex = torch.sigmoid(self.gate_tex[i])
            g_str = torch.sigmoid(self.gate_str[i])

            ctx_tex = self._resize_to_context(F_past_up[i], cur)
            ctx_str = self._resize_to_context(F_blur_up[i], cur)

            tex = self.cross_attn_up_tex[i](cur, ctx_tex)
            strc = self.cross_attn_up_str[i](cur, ctx_str)
            cur = cur + g_tex * tex + g_str * strc

            cur = upsample(cur)

        # head
        if cond_full.shape[2:] != cur.shape[2:]:
            cond_full_resized = F.interpolate(cond_full, size=cur.shape[2:], mode="trilinear", align_corners=False)
        else:
            cond_full_resized = cond_full

        cur = torch.cat((cur, r, cond_full_resized), dim=1)
        cur = self.final_res_block(cur, t_emb)
        return self.final_conv(cur)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Conditional Diffusion"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 file (N,T,H,W,3)")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--results", type=str, default="./results_video")
    args = parser.parse_args()

    unet3d = Unet3D(
        dim=32,
        init_dim=None,
        attn_type=["time","space","full","full"],
        cross_attn=[False, False, True, True],
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