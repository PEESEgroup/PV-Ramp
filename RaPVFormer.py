import argparse
import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utilities import (
    HybridVideoDataset,
    PAPER_SPLIT_SEED,
    move_batch_to_device,
    save_paper_checkpoint,
    save_hyperparam,
    select_device,
    set_random_seed,
    setup_logger,
    split_dataset_by_date,
    split_manifest_hash,
    validate_manifest_hash,
)


def slope_loss(pred, target, asymmetric=1.0):
    if pred.ndim != 2 or target.ndim != 2 or pred.shape != target.shape:
        raise ValueError("pred and target must have identical shape [B,T].")
    if pred.shape[1] < 2:
        raise ValueError("slope_loss requires at least two forecast steps.")
    predicted_slope = pred[:, 1:] - pred[:, :-1]
    target_slope = target[:, 1:] - target[:, :-1]
    median = target_slope.abs().median(dim=1, keepdim=True).values
    weights = 1.0 + asymmetric * torch.sigmoid(target_slope.abs() - median)
    return (((predicted_slope - target_slope) ** 2) * weights).mean()


def ramp_targets(target, previous_pv, threshold):
    """Return down/stable/up labels (0/1/2), including issuance-to-lead-1."""
    if target.ndim != 2 or previous_pv.ndim != 1 or target.shape[0] != previous_pv.shape[0]:
        raise ValueError("target must be [B,T] and previous_pv must be [B].")
    extended = torch.cat([previous_pv[:, None], target], dim=1)
    increments = extended[:, 1:] - extended[:, :-1]
    return torch.where(
        increments > threshold,
        torch.full_like(increments, 2, dtype=torch.long),
        torch.where(
            increments < -threshold,
            torch.zeros_like(increments, dtype=torch.long),
            torch.ones_like(increments, dtype=torch.long),
        ),
    )


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        batch, steps, classes = logits.shape
        ce = F.cross_entropy(logits.reshape(-1, classes), targets.reshape(-1), reduction="none")
        with torch.no_grad():
            probability = torch.softmax(logits, dim=-1).reshape(-1, classes)
            pt = probability.gather(1, targets.reshape(-1, 1)).clamp_min(1e-6).squeeze(1)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class ConvEncoder(nn.Module):
    def __init__(self, in_ch=4, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, dim, 3, stride=2, padding=1), nn.BatchNorm2d(dim), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.net(x)).flatten(1)


class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        return x + self.encoding[: x.shape[1]].unsqueeze(0).to(dtype=x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, dim_feedforward), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attended = self.attention(x, x, x, need_weights=False)[0]
        x = self.norm1(x + attended)
        return self.norm2(x + self.feedforward(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 4 * d_model), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, context):
        attended = self.attention(query, context, context, need_weights=False)[0]
        x = self.norm1(query + attended)
        return self.norm2(x + self.feedforward(x))


class RaPVFormer(nn.Module):
    def __init__(self, d_img=128, d_model=256, nhead=4, depth_hist=2,
                 depth_fut=2, depth_fuse=2, pv_ctx=16, cap: Optional[float] = 30.1):
        super().__init__()
        self.enc_hist = ConvEncoder(4, d_img)
        self.enc_fut = ConvEncoder(4, d_img)
        self.pv_embed = nn.Sequential(nn.Linear(1, pv_ctx), nn.SiLU(), nn.Linear(pv_ctx, pv_ctx))
        self.hist_proj = nn.Linear(d_img + pv_ctx, d_model)
        self.fut_proj = nn.Linear(d_img, d_model)
        self.pos = TimePositionalEncoding(d_model)
        self.hist_blocks = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(depth_hist)])
        self.fut_blocks = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(depth_fut)])
        self.fuse_blocks = nn.ModuleList([CrossAttentionBlock(d_model, nhead) for _ in range(depth_fuse)])
        self.head_pv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.head_ramp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 3),
        )
        self.cap = cap

    @staticmethod
    def _validate_inputs(past_rgb, past_sun, past_pv, future_rgb, future_sun):
        expected = (16,)
        if past_rgb.ndim != 5 or past_rgb.shape[1:3] != (16, 3):
            raise ValueError("past_rgb must have shape [B,16,3,H,W].")
        if future_rgb.ndim != 5 or future_rgb.shape[1:3] != (16, 3):
            raise ValueError("future_rgb must have shape [B,16,3,H,W].")
        if past_sun.shape[1:3] != (16, 1) or future_sun.shape[1:3] != (16, 1):
            raise ValueError("Sun masks must have shape [B,16,1,H,W].")
        if past_pv.ndim != 2 or past_pv.shape[1:] != expected:
            raise ValueError("past_pv must have shape [B,16].")
        tensors = (past_sun, future_rgb, future_sun)
        if any(t.shape[0] != past_rgb.shape[0] or t.shape[-2:] != past_rgb.shape[-2:] for t in tensors):
            raise ValueError("All image inputs must share batch and spatial dimensions.")

    @staticmethod
    def _encode_images(encoder, images):
        batch, steps, channels, height, width = images.shape
        encoded = encoder(images.reshape(batch * steps, channels, height, width))
        return encoded.reshape(batch, steps, -1)

    def forward(self, past_rgb, past_sun, past_pv, future_rgb, future_sun) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(past_rgb, past_sun, past_pv, future_rgb, future_sun)
        historical_images = torch.cat([past_rgb, past_sun], dim=2)
        future_images = torch.cat([future_rgb, future_sun], dim=2)
        historical = self._encode_images(self.enc_hist, historical_images)
        future = self._encode_images(self.enc_fut, future_images)
        pv_embedding = self.pv_embed(past_pv.unsqueeze(-1))
        historical = self.pos(self.hist_proj(torch.cat([historical, pv_embedding], dim=-1)))
        future = self.pos(self.fut_proj(future))
        for block in self.hist_blocks:
            historical = block(historical)
        for block in self.fut_blocks:
            future = block(future)
        for block in self.fuse_blocks:
            future = block(future, historical)
        pv_logits = self.head_pv(future).squeeze(-1)
        predicted_pv = torch.relu(pv_logits) * self.cap if self.cap is not None else pv_logits
        return predicted_pv, self.head_ramp(future)


class RampCompositeLoss(nn.Module):
    def __init__(self, w_pv=1.0, w_s=0.2, w_r=0.5, capacity=30.1,
                 noise_tolerance=0.05, focal_gamma=2.0, asymmetric=1.0):
        super().__init__()
        self.w_pv, self.w_s, self.w_r = w_pv, w_s, w_r
        self.class_threshold = capacity * noise_tolerance
        self.focal = FocalLoss(gamma=focal_gamma)
        self.asymmetric = asymmetric

    def components(self, predicted_pv, ramp_logits, target, previous_pv):
        mse = F.mse_loss(predicted_pv, target)
        slope = slope_loss(predicted_pv, target, asymmetric=self.asymmetric)
        labels = ramp_targets(target, previous_pv, self.class_threshold)
        focal = self.focal(ramp_logits, labels)
        total = self.w_pv * mse + self.w_s * slope + self.w_r * focal
        return total, {"mse": mse, "slope": slope, "focal": focal}

    def forward(self, predicted_pv, ramp_logits, target, previous_pv):
        return self.components(predicted_pv, ramp_logits, target, previous_pv)[0]


def dual_view_loss(model, criterion, batch):
    common = (batch["past_rgb"], batch["past_sun"], batch["past_pv"])
    real_pv, real_ramp = model(*common, batch["future_rgb"], batch["future_sun"])
    generated_pv, generated_ramp = model(
        *common, batch["future_generated_rgb"], batch["future_sun"]
    )
    previous = batch["past_pv"][:, -1]
    real_loss = criterion(real_pv, real_ramp, batch["future_pv"], previous)
    generated_loss = criterion(generated_pv, generated_ramp, batch["future_pv"], previous)
    return 0.5 * (real_loss + generated_loss)


@torch.no_grad()
def evaluate_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = total_mse = 0.0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        predicted, logits = model(
            batch["past_rgb"], batch["past_sun"], batch["past_pv"],
            batch["future_generated_rgb"], batch["future_sun"],
        )
        loss, components = criterion.components(
            predicted, logits, batch["future_pv"], batch["past_pv"][:, -1]
        )
        total_loss += float(loss)
        total_mse += float(components["mse"])
    count = max(1, len(loader))
    return total_loss / count, total_mse / count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train paper-aligned RaPVFormer")
    parser.add_argument("--data", default=os.path.join("data", "video_frames.h5"))
    parser.add_argument("--predictions", default=os.path.join("results", "phydiffnet_predictions.h5"))
    parser.add_argument("--split_manifest", default=os.path.join("data", "split_manifest.json"))
    parser.add_argument("-m", "--model_name", default="RaPVFormer")
    parser.add_argument("-id", "--model_id", type=int, default=0)
    parser.add_argument("-r", "--random_seed", type=int, default=PAPER_SPLIT_SEED)
    parser.add_argument("-bz", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-e", "--num_epochs", type=int, default=30)
    parser.add_argument("--w_pv", type=float, default=1.0)
    parser.add_argument("--w_s", type=float, default=0.2)
    parser.add_argument("--w_r", type=float, default=0.5)
    parser.add_argument("--noise_tolerance", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--slope_asym", type=float, default=1.0)
    parser.add_argument("--pv_capacity", type=float, default=30.1)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--save_model_every_epoch", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logger(f"RaPVFormer_{args.model_id:02}", args)
    save_hyperparam(args)
    set_random_seed(args.random_seed)
    device = select_device()
    dataset = HybridVideoDataset(args.data, args.predictions)
    train_set, val_set, test_set, manifest = split_dataset_by_date(
        dataset, args.split_manifest, seed=args.random_seed
    )
    prediction_hash = dataset.prediction_metadata.get("split_manifest_hash")
    validate_manifest_hash(
        prediction_hash, split_manifest_hash(manifest), "PhyDiffNet predictions"
    )
    loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_set, batch_size=args.batch_size),
        "test": DataLoader(test_set, batch_size=args.batch_size),
    }
    model = RaPVFormer(cap=args.pv_capacity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.3)
    criterion = RampCompositeLoss(
        args.w_pv, args.w_s, args.w_r, args.pv_capacity,
        args.noise_tolerance, args.focal_gamma, args.slope_asym,
    )
    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location=device, weights_only=False)
        if checkpoint.get("architecture_version") != "paper-v1":
            raise ValueError("Only versioned paper-v1 RaPVFormer checkpoints are accepted.")
        validate_manifest_hash(
            checkpoint.get("split_manifest_hash"),
            split_manifest_hash(manifest),
            "RaPVFormer checkpoint",
        )
        model.load_state_dict(checkpoint["model"], strict=True)

    weights_dir = os.path.join("weights", f"{args.model_name}_{args.model_id:02}")
    for epoch in range(args.num_epochs):
        model.train()
        running = 0.0
        for batch in loaders["train"]:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = dual_view_loss(model, criterion, batch)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        val_loss, val_mse = evaluate_epoch(model, loaders["val"], device, criterion)
        scheduler.step(val_loss)
        logging.info(
            "Epoch %d/%d train=%.6f val=%.6f val_mse=%.6f",
            epoch + 1, args.num_epochs, running / max(1, len(loaders["train"])), val_loss, val_mse,
        )
        if (epoch + 1) % args.save_model_every_epoch == 0:
            save_paper_checkpoint(
                os.path.join(weights_dir, f"pv_ramp_net_{epoch + 1:03}.pth"),
                model, optimizer, args, manifest, epoch + 1,
            )
    test_loss, test_mse = evaluate_epoch(model, loaders["test"], device, criterion)
    logging.info("Final test loss=%.6f mse=%.6f", test_loss, test_mse)
    save_paper_checkpoint(
        os.path.join(weights_dir, "pv_ramp_net.pth"),
        model, optimizer, args, manifest, args.num_epochs,
    )


if __name__ == "__main__":
    main()
