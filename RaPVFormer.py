import os
import math
import logging
import argparse
from typing import Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utilities import *

# -----------------------------
# Low-level utils / losses
# -----------------------------

def causal_mask(T_q: int, T_k: int, device: torch.device):
    m = torch.ones(T_q, T_k, device=device).tril()
    return m == 0


def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]):
    diff = target.unsqueeze(-1) - pred_q
    losses = []
    for i, q in enumerate(quantiles):
        e = diff[..., i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.mean(torch.stack(losses, dim=-1))


def slope_loss(pred: torch.Tensor, target: torch.Tensor, asymmetric: float = 1.0):
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    if dt.numel() == 0:
        return pred.new_tensor(0.0)
    scale = dt.abs()
    w = 1.0 + asymmetric * torch.sigmoid(scale - scale.median())
    return torch.mean(((dp - dt) ** 2) * w)


def ramp_targets(target: torch.Tensor, thr: float) -> torch.Tensor:
    dt = target[:, 1:] - target[:, :-1]
    y = torch.zeros_like(target, dtype=torch.long)
    y[:, 1:] = torch.where(dt > thr, 1, torch.where(dt < -thr, -1, 0)) + 1
    return y


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, T, C = logits.shape
        ce = F.cross_entropy(logits.view(-1, C), targets.view(-1), reduction="none")
        with torch.no_grad():
            p = torch.softmax(logits, dim=-1).view(-1, C)
            pt = p.gather(1, targets.view(-1, 1)).clamp_min(1e-6)
        loss = ((1 - pt) ** self.gamma).view_as(ce) * ce
        return loss.mean()


# -----------------------------
# Backbone blocks
# -----------------------------

class ConvEncoder(nn.Module):
    def __init__(self, in_ch=4, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        h = self.net(x)
        h = self.pool(h).flatten(1)
        return h


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        h = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False
        )[0]
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, key_padding_mask=None, attn_mask=None):
        h = self.attn(
            q, kv, kv, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False
        )[0]
        x = self.norm1(q + h)
        x = self.norm2(x + self.ff(x))
        return x


# -----------------------------
# Ramp-aware PV model
# -----------------------------

class RampAwarePVNet(nn.Module):
    """History+future image encoder + PV history encoding."""

    def __init__(
        self,
        img_channels=4,
        d_img=128,
        d_model=256,
        nhead=4,
        depth_hist=2,
        depth_fut=2,
        depth_fuse=2,
        pv_ctx=16,
        cap: Optional[float] = None,
    ):
        super().__init__()
        self.enc_hist = ConvEncoder(img_channels, d_img)
        self.enc_fut = ConvEncoder(img_channels, d_img)

        self.pv_embed = nn.Sequential(
            nn.Linear(1, pv_ctx), nn.SiLU(), nn.Linear(pv_ctx, pv_ctx)
        )

        self.hist_proj = nn.Linear(d_img + pv_ctx, d_model)
        self.fut_proj = nn.Linear(d_img, d_model)

        self.pos = TimePositionalEncoding(d_model)

        self.hist_blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead) for _ in range(depth_hist)]
        )
        self.fut_blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead) for _ in range(depth_fut)]
        )
        self.fuse_blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, nhead) for _ in range(depth_fuse)]
        )

        self.head_pv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.head_ramp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),
        )

        self.cap = cap

    @staticmethod
    def _encode_images(enc: ConvEncoder, imgs: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = imgs.shape
        x = imgs.view(b * t, c, h, w)
        z = enc(x)
        return z.view(b, t, -1)

    def forward(
        self,
        past_imgs: torch.Tensor,
        past_pv: torch.Tensor,
        future_imgs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t_h = past_pv.shape
        t_f = future_imgs.shape[1]

        z_hist = self._encode_images(self.enc_hist, past_imgs)
        z_fut = self._encode_images(self.enc_fut, future_imgs)

        pv_emb = self.pv_embed(past_pv.unsqueeze(-1))
        hist = torch.cat([z_hist, pv_emb], dim=-1)
        hist = self.hist_proj(hist)
        fut = self.fut_proj(z_fut)

        hist = self.pos(hist)
        fut = self.pos(fut)

        for blk in self.hist_blocks:
            hist = blk(hist)
        for blk in self.fut_blocks:
            fut = blk(fut)
        for blk in self.fuse_blocks:
            fut = blk(fut, hist)

        pv = self.head_pv(fut)             # [B, T_f, 1]
        ramp_logits = self.head_ramp(fut)   # [B, T_f, 3]

        if self.cap is not None:
            cap_tensor = torch.tensor([self.cap], device=hist.device).view(-1, 1, 1)
            pv = torch.relu(pv) * cap_tensor

        return pv, ramp_logits


# -----------------------------
# Composite training loss
# -----------------------------

class RampCompositeLoss(nn.Module):
    def __init__(
        self,
        w_pv: float = 1.0,
        w_s: float = 0.2,
        w_r: float = 0.5,
        ramp_thr: float = 0.02,
        focal_gamma: float = 2.0,
        asymmetric: float = 1.0,
    ):
        super().__init__()
        self.w_pv = w_pv
        self.w_s = w_s
        self.w_r = w_r
        self.ramp_thr = ramp_thr
        self.focal = FocalLoss(gamma=focal_gamma)
        self.asymmetric = asymmetric

    def forward(self, pred_pv: torch.Tensor, ramp_logits: torch.Tensor, target: torch.Tensor):
        if pred_pv.dim() == 3:
            pred = pred_pv.squeeze(-1)
        else:
            pred = pred_pv

        if target.dim() == 3:
            target = target.squeeze(-1)

        loss_pv = F.mse_loss(pred, target)
        loss_s = slope_loss(pred, target, asymmetric=self.asymmetric)

        labels = ramp_targets(target, thr=self.ramp_thr)
        loss_r = self.focal(ramp_logits, labels)

        return self.w_pv * loss_pv + self.w_s * loss_s + self.w_r * loss_r


class PvInterfaceModel(nn.Module):
    def __init__(self, cap: Optional[float]):
        super().__init__()
        self.core = RampAwarePVNet(cap=cap)

    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
        input_pv: torch.Tensor,
    ) -> torch.Tensor:
        past_pv = input_pv.squeeze(-1)
        pv, _ = self.core(
            past_imgs=input_frames,
            past_pv=past_pv,
            future_imgs=target_frames,
        )
        return pv  # [B, T_f, 1]

    def predict_all(
        self,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
        input_pv: torch.Tensor,
    ):
        past_pv = input_pv.squeeze(-1)
        return self.core(
            past_imgs=input_frames,
            past_pv=past_pv,
            future_imgs=target_frames,
        )


def evaluate_epoch(model, loader, device, criterion, mse_loss):
    model.eval()
    tot_loss, tot_mse = 0.0, 0.0
    with torch.no_grad():
        for _, loaddata in enumerate(loader, 0):
            loaddata = [data.to(device) for data in loaddata]
            _, input_frames, target_frames, input_pv, target_pv = loaddata
            pred_pv, ramp_logits = model.predict_all(input_frames, target_frames, input_pv)
            loss = criterion(pred_pv, ramp_logits, target_pv)
            tot_loss += float(loss)
            tot_mse += float(mse_loss(pred_pv, target_pv))
    n = max(1, len(loader))
    return tot_loss / n, tot_mse / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="RaPVFormer")
    parser.add_argument("-id", "--model_id", type=int, default=0)

    parser.add_argument("-r", "--random_seed", type=int, default=42)
    parser.add_argument("-bz", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-e", "--num_epochs", type=int, default=100)
    parser.add_argument("-strd", "--stride", type=int, default=1)

    parser.add_argument("-df", "--disp_every_batch", type=int, default=100)
    parser.add_argument("-smd", "--save_model_every_epoch", type=int, default=5)
    parser.add_argument("-ldm", "--load_model", type=str, default="")
    parser.add_argument("-msg", "--message", type=str, default="")

    parser.add_argument("--w_pv", type=float, default=1.0)
    parser.add_argument("--w_s", type=float, default=0.2)
    parser.add_argument("--w_r", type=float, default=0.5)
    parser.add_argument("--ramp_thr", type=float, default=30.1 * 0.2)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--slope_asym", type=float, default=1.0)

    parser.add_argument("-cap", "--pv_capacity", type=float, default=30.1)

    args = parser.parse_args()

    if os.name == "nt":
        args.batch_size = 4
        args.disp_every_batch = 1

    script_name = os.path.basename(__file__).split(".")[0] + f"_{args.model_id:02}"
    setup_logger(script_name, args)
    save_hyperparam(args)
    set_random_seed(args.random_seed)
    device = select_device()

    dataset = SkyVideoDataset(stride=args.stride)

    train_dataset, val_dataset, test_dataset = split_dataset_by_date(dataset)
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Val   dataset size: {len(val_dataset)}")
    logging.info(f"Test  dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    model = PvInterfaceModel(cap=args.pv_capacity).to(device)

    if args.load_model:
        try:
            state = torch.load(args.load_model, map_location=device)
            model.load_state_dict(state, strict=False)
            logging.info(f"Loaded model weights from {args.load_model}")
        except Exception as ex:
            logging.warning(f"Failed to load weights: {ex}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.3)
    mse = nn.MSELoss()

    crit = RampCompositeLoss(
        w_pv=args.w_pv,
        w_s=args.w_s,
        w_r=args.w_r,
        ramp_thr=args.ramp_thr,
        focal_gamma=args.focal_gamma,
        asymmetric=args.slope_asym,
    )

    for epoch_idx in range(0, args.num_epochs):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0

        for batch_idx, loaddata in enumerate(train_loader, 0):
            loaddata = [data.to(device) for data in loaddata]
            _, input_frames, target_frames, input_pv, target_pv = loaddata

            optimizer.zero_grad()
            pred_pv, ramp_logits = model.predict_all(input_frames, target_frames, input_pv)
            loss = crit(pred_pv, ramp_logits, target_pv)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_mse += float(mse(pred_pv, target_pv).detach())

            if batch_idx == 0 or (batch_idx + 1) % args.disp_every_batch == 0:
                logging.info(
                    f"Epoch [{epoch_idx+1}/{args.num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss:.6f}  MSE: {mse(pred_pv, target_pv):.6f}"
                )

        n_train = max(1, len(train_loader))
        logging.info(
            f"Epoch [{epoch_idx+1}/{args.num_epochs}] "
            f"Train Loss: {epoch_loss/n_train:.6f}  Train MSE: {epoch_mse/n_train:.6f}"
        )

        # ---- Val ----
        val_loss, val_mse = evaluate_epoch(model, val_loader, device, crit, mse)
        logging.info(
            f"Epoch [{epoch_idx+1}/{args.num_epochs}] "
            f"Val   Loss: {val_loss:.6f}  Val   MSE: {val_mse:.6f}"
        )
        scheduler.step(val_loss)

        # ---- Save ----
        if (epoch_idx + 1) % args.save_model_every_epoch == 0:
            model_save_path = os.path.join(
                os.path.dirname(__file__), "weights", f"{args.model_name}_{args.model_id:02}"
            )
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, f"pv_ramp_net_{(epoch_idx+1):03}.pth"),
            )
            torch.save(model.state_dict(), os.path.join(model_save_path, "pv_ramp_net.pth"))
            logging.info(f"Saved model to {model_save_path}")

    # ---- Final Test ----
    test_loss, test_mse = evaluate_epoch(model, test_loader, device, crit, mse)
    logging.info(f"[Final] Test Loss: {test_loss:.6f}  Test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
