import argparse
import os

import h5py
import torch
from torch.utils.data import DataLoader

from utilities import (
    HybridVideoDataset,
    PAPER_SPLIT_SEED,
    load_split_manifest,
    normalize_to_neg_one_to_one,
    select_device,
    set_random_seed,
    unnormalize_to_zero_to_one,
    validate_manifest_hash,
    video_tensor_to_uint8,
)
from video_conditional_diffusion import build_diffusion


@torch.no_grad()
def generate(dataset, diffusion, output, *, batch_size=8, seed=PAPER_SPLIT_SEED,
             manifest_hash=None):
    device = next(diffusion.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    generator = torch.Generator(device=device).manual_seed(seed)
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    offset = 0
    with h5py.File(output, "w") as handle:
        prediction_dataset = None
        index_dataset = handle.create_dataset(
            "eval_stidx", (len(dataset),), dtype="i8", compression="lzf"
        )
        for batch in loader:
            past = batch["past_rgb"][:, -4:].permute(0, 2, 1, 3, 4).to(device)
            coarse = batch["future_generated_rgb"].permute(0, 2, 1, 3, 4).to(device)
            refined = diffusion.sample(
                normalize_to_neg_one_to_one(past),
                normalize_to_neg_one_to_one(coarse),
                generator=generator,
            )
            refined = unnormalize_to_zero_to_one(refined).clamp(0, 1)
            refined = video_tensor_to_uint8(refined.permute(0, 2, 1, 3, 4))
            if prediction_dataset is None:
                prediction_dataset = handle.create_dataset(
                    "prediction_imgs", (len(dataset), *refined.shape[1:]),
                    dtype="u1", compression="lzf",
                )
            count = refined.shape[0]
            prediction_dataset[offset : offset + count] = refined
            index_dataset[offset : offset + count] = batch["start_idx"].numpy()
            offset += count
        handle.attrs["architecture_version"] = "paper-v1"
        handle.attrs["split_manifest_hash"] = manifest_hash or ""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate refined PhyDiffNet clips")
    parser.add_argument("--data", default=os.path.join("data", "video_frames.h5"))
    parser.add_argument("--coarse_predictions", default=os.path.join("results", "phydnet_coarse_predictions.h5"))
    parser.add_argument("--diffusion_checkpoint", required=True)
    parser.add_argument("--output", default=os.path.join("results", "phydiffnet_predictions.h5"))
    parser.add_argument("--split_manifest", default=os.path.join("data", "split_manifest.json"))
    parser.add_argument("--architecture", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=PAPER_SPLIT_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_random_seed(args.seed)
    device = select_device()
    diffusion = build_diffusion(args.architecture).to(device)
    checkpoint = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)
    saved_architecture = checkpoint.get("architecture_version")
    if saved_architecture is None and args.architecture != "legacy":
        raise ValueError("Unversioned diffusion checkpoints require --architecture legacy.")
    if saved_architecture is not None and saved_architecture != args.architecture:
        raise ValueError("Checkpoint and requested diffusion architectures differ.")
    diffusion.load_state_dict(checkpoint["model"], strict=True)

    _, manifest_hash = load_split_manifest(args.split_manifest)
    validate_manifest_hash(
        checkpoint.get("split_manifest_hash"),
        manifest_hash,
        "Diffusion checkpoint",
        allow_missing=True,
    )
    dataset = HybridVideoDataset(args.data, args.coarse_predictions)
    validate_manifest_hash(
        dataset.prediction_metadata.get("split_manifest_hash"),
        manifest_hash,
        "Coarse predictions",
        allow_missing=True,
    )
    generate(
        dataset, diffusion, args.output, batch_size=args.batch_size,
        seed=args.seed, manifest_hash=manifest_hash,
    )


if __name__ == "__main__":
    main()
