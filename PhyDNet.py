import argparse
import logging
import math
import os
import random

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from constrain_moments import K2M
from rnn_models import ConvLSTM, FrameLoss, PhyCell, PhyDNet as PhyDNetModel
from utilities import (
    DatasetView,
    PAPER_SPLIT_SEED,
    SkyVideoDataset,
    count_parameters,
    move_batch_to_device,
    save_paper_checkpoint,
    save_hyperparam,
    select_device,
    set_random_seed,
    setup_logger,
    split_dataset_by_date,
    split_manifest_hash,
    validate_manifest_hash,
    write_prediction_hdf5,
)


SAMPLING_STEP_1 = 15
SAMPLING_STEP_2 = 30
REVERSE_EXP_ALPHA = 2.5


def reserve_schedule_sampling_exp(epoch_idx, length):
    if epoch_idx < SAMPLING_STEP_1:
        ratio = 0.5
    elif epoch_idx < SAMPLING_STEP_2:
        ratio = 1.0 - 0.5 * math.exp(-(epoch_idx - SAMPLING_STEP_1) / REVERSE_EXP_ALPHA)
    else:
        ratio = 1.0
    return ratio, [random.random() < ratio for _ in range(length)]


def schedule_sampling(epoch_idx, length):
    if epoch_idx < SAMPLING_STEP_1:
        ratio = 0.5
    elif epoch_idx < SAMPLING_STEP_2:
        ratio = 0.5 - 0.5 * (epoch_idx - SAMPLING_STEP_1) / (SAMPLING_STEP_2 - SAMPLING_STEP_1)
    else:
        ratio = 0.0
    return ratio, [random.random() < ratio for _ in range(length)]


def build_model(device, hidden_dim1=128, hidden_dim2=128):
    phycell = PhyCell(
        input_shape=(32, 32), input_dim=64, F_hidden_dims=[49],
        n_layers=1, kernel_size=(7, 7), device=device,
    )
    convcell = ConvLSTM(
        input_shape=(32, 32), input_dim=64,
        hidden_dims=[hidden_dim1, hidden_dim2, 64],
        n_layers=3, kernel_size=(3, 3), device=device,
    )
    return PhyDNetModel(phycell, convcell, device).to(device)


def build_moment_targets(device):
    return torch.eye(49, device=device).reshape(49, 7, 7)


def moment_constraint_loss(model, converter, targets, mse):
    """Original PhyDNet differential-moment constraint, exposed explicitly."""
    total = targets.new_tensor(0.0)
    weights = model.phycell.cell_list[0].F.conv1.weight
    for channel in range(model.phycell.cell_list[0].input_dim):
        moments = converter(weights[:, channel].double()).float()
        total = total + mse(moments, targets)
    return total


def train_on_batch(batch, epoch_idx, model, optimizer, frame_criterion, converter,
                   moment_targets, lambda_phy):
    model.train()
    batch = move_batch_to_device(batch, next(model.parameters()).device)
    optimizer.zero_grad()
    _, encoder_flags = reserve_schedule_sampling_exp(epoch_idx, 16)
    _, decoder_flags = schedule_sampling(epoch_idx, 16)

    frame_total = batch["past_rgb"].new_tensor(0.0)
    ssim_total = frame_total.clone()
    l1_total = frame_total.clone()
    current = torch.cat([batch["past_rgb"][:, 0], batch["past_sun"][:, 0]], dim=1)

    for index in range(15):
        prediction = model(current, first_timestep=index == 0)
        loss, ssim_value, l1_value = frame_criterion.components(
            prediction, batch["past_rgb"][:, index + 1]
        )
        frame_total = frame_total + loss
        ssim_total = ssim_total + ssim_value
        l1_total = l1_total + l1_value
        rgb = batch["past_rgb"][:, index + 1] if encoder_flags[index] else prediction
        current = torch.cat([rgb, batch["past_sun"][:, index + 1]], dim=1)

    if encoder_flags[-1]:
        current = torch.cat([batch["past_rgb"][:, -1], batch["past_sun"][:, -1]], dim=1)

    for index in range(16):
        prediction = model(current)
        loss, ssim_value, l1_value = frame_criterion.components(
            prediction, batch["future_rgb"][:, index]
        )
        frame_total = frame_total + loss
        ssim_total = ssim_total + ssim_value
        l1_total = l1_total + l1_value
        rgb = batch["future_rgb"][:, index] if decoder_flags[index] else prediction
        current = torch.cat([rgb, batch["future_sun"][:, index]], dim=1)

    frame_total = frame_total / 31.0
    ssim_total = ssim_total / 31.0
    l1_total = l1_total / 31.0
    physical = moment_constraint_loss(model, converter, moment_targets, nn.functional.mse_loss)
    total = frame_total + lambda_phy * physical
    total.backward()
    optimizer.step()
    return {
        "total": float(total.detach()),
        "frame": float(frame_total.detach()),
        "ssim": float(ssim_total.detach()),
        "l1": float(l1_total.detach()),
        "physical": float(physical.detach()),
    }


@torch.no_grad()
def evaluate(model, loader, frame_criterion):
    model.eval()
    device = next(model.parameters()).device
    predictions, start_indices = [], []
    losses = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        predicted = model.predict_sequence(
            batch["past_rgb"], batch["past_sun"], batch["future_sun"]
        )
        loss = torch.stack([
            frame_criterion(predicted[:, i], batch["future_rgb"][:, i])
            for i in range(16)
        ]).mean()
        losses.append(float(loss))
        predictions.append(predicted.cpu())
        start_indices.append(batch["start_idx"].cpu())
    if not predictions:
        raise ValueError("Cannot evaluate an empty dataset split.")
    return torch.cat(predictions), torch.cat(start_indices), sum(losses) / len(losses)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train paper-aligned PhyDNet")
    parser.add_argument("--data", default=os.path.join("data", "video_frames.h5"))
    parser.add_argument("--split_manifest", default=os.path.join("data", "split_manifest.json"))
    parser.add_argument("--predictions", default=os.path.join("results", "phydnet_coarse_predictions.h5"))
    parser.add_argument("-m", "--model_name", default="PhyDNet")
    parser.add_argument("-id", "--model_id", type=int, default=0)
    parser.add_argument("-r", "--random_seed", type=int, default=PAPER_SPLIT_SEED)
    parser.add_argument("-bz", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-e", "--num_epochs", type=int, default=300)
    parser.add_argument("-hd1", "--hidden_dim1", type=int, default=128)
    parser.add_argument("-hd2", "--hidden_dim2", type=int, default=128)
    parser.add_argument("--lambda_phy", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.84)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--save_model_every_epoch", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logger(f"PhyDNet_{args.model_id:02}", args)
    save_hyperparam(args)
    set_random_seed(args.random_seed)
    device = select_device(req_mem=5000)
    dataset = SkyVideoDataset(hdf5_path=args.data)
    train_set, val_set, test_set, manifest = split_dataset_by_date(
        dataset, args.split_manifest, seed=args.random_seed
    )
    loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_set, batch_size=args.batch_size),
        "test": DataLoader(test_set, batch_size=args.batch_size),
    }
    model = build_model(device, args.hidden_dim1, args.hidden_dim2)
    logging.info("Trainable parameters: %d", count_parameters(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.3)
    frame_criterion = FrameLoss(alpha=args.alpha).to(device)
    converter = K2M([7, 7]).to(device)
    moment_targets = build_moment_targets(device)

    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location=device, weights_only=False)
        if "model" in checkpoint:
            validate_manifest_hash(
                checkpoint.get("split_manifest_hash"),
                split_manifest_hash(manifest),
                "PhyDNet checkpoint",
                allow_missing=True,
            )
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)

    weights_dir = os.path.join("weights", f"{args.model_name}_{args.model_id:02}")
    for epoch in range(args.num_epochs):
        metrics = []
        for batch in loaders["train"]:
            metrics.append(train_on_batch(
                batch, epoch, model, optimizer, frame_criterion,
                converter, moment_targets, args.lambda_phy,
            ))
        val_result = evaluate(model, loaders["val"], frame_criterion)
        scheduler.step(val_result[2])
        count = max(1, len(metrics))
        means = {
            name: sum(item[name] for item in metrics) / count
            for name in ("total", "frame", "ssim", "l1", "physical")
        }
        logging.info(
            "Epoch %d/%d total=%.6f frame=%.6f ssim=%.6f l1=%.6f physical=%.6f val=%.6f",
            epoch + 1, args.num_epochs, means["total"], means["frame"],
            means["ssim"], means["l1"], means["physical"], val_result[2],
        )
        if (epoch + 1) % args.save_model_every_epoch == 0:
            save_paper_checkpoint(
                os.path.join(weights_dir, f"encoder_{epoch + 1:03}.pth"),
                model, optimizer, args, manifest, epoch + 1,
            )

    prediction_loaders = {
        "train": DataLoader(DatasetView(dataset, train_set.indices), batch_size=args.batch_size),
        "val": loaders["val"],
        "test": loaders["test"],
    }
    results = {name: evaluate(model, loader, frame_criterion) for name, loader in prediction_loaders.items()}
    prediction_frames = torch.cat([results[name][0] for name in ("train", "val", "test")])
    prediction_indices = torch.cat([results[name][1] for name in ("train", "val", "test")])
    write_prediction_hdf5(
        args.predictions,
        prediction_frames,
        prediction_indices,
        manifest_hash=split_manifest_hash(manifest),
    )
    save_paper_checkpoint(
        os.path.join(weights_dir, "encoder.pth"),
        model, optimizer, args, manifest, args.num_epochs,
    )


if __name__ == "__main__":
    main()
