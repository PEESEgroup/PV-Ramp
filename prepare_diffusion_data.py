import argparse
import json
import os

import h5py

from utilities import (
    HybridVideoDataset,
    PAPER_SPLIT_SEED,
    build_or_load_day_split,
    split_manifest_hash,
    validate_manifest_hash,
    video_tensor_to_uint8,
)


def prepare(base_data, coarse_predictions, output, split_manifest, seed=PAPER_SPLIT_SEED):
    dataset = HybridVideoDataset(base_data, coarse_predictions)
    indices, manifest = build_or_load_day_split(dataset, split_manifest, seed=seed)
    expected_hash = split_manifest_hash(manifest)
    prediction_hash = dataset.prediction_metadata.get("split_manifest_hash")
    validate_manifest_hash(
        prediction_hash, expected_hash, "Coarse predictions", allow_missing=True
    )
    train_indices = indices["train"]
    if not train_indices:
        raise ValueError("Training split is empty.")
    first = dataset.get_item(train_indices[0], augment=False)
    height, width = first["past_rgb"].shape[-2:]
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with h5py.File(output, "w") as handle:
        kwargs = {"compression": "lzf"}
        real = handle.create_dataset("real_vids", (len(train_indices), 16, height, width, 3), dtype="u1", **kwargs)
        pred = handle.create_dataset("pred_vids", (len(train_indices), 16, height, width, 3), dtype="u1", **kwargs)
        prev = handle.create_dataset("prev_vids", (len(train_indices), 4, height, width, 3), dtype="u1", **kwargs)
        starts = handle.create_dataset("start_idx", (len(train_indices),), dtype="i8", **kwargs)
        for output_index, dataset_index in enumerate(train_indices):
            item = dataset.get_item(dataset_index, augment=False)
            real[output_index] = video_tensor_to_uint8(item["future_rgb"])
            pred[output_index] = video_tensor_to_uint8(item["future_generated_rgb"])
            prev[output_index] = video_tensor_to_uint8(item["past_rgb"][-4:])
            starts[output_index] = int(item["start_idx"])
        handle.attrs["split"] = "train"
        handle.attrs["split_manifest_hash"] = expected_hash
        handle.attrs["split_manifest"] = json.dumps(manifest, sort_keys=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the train-only conditional-diffusion HDF5 file")
    parser.add_argument("--data", default=os.path.join("data", "video_frames.h5"))
    parser.add_argument("--coarse_predictions", default=os.path.join("results", "phydnet_coarse_predictions.h5"))
    parser.add_argument("--output", default=os.path.join("data", "diffusion_train.h5"))
    parser.add_argument("--split_manifest", default=os.path.join("data", "split_manifest.json"))
    parser.add_argument("--split_seed", type=int, default=PAPER_SPLIT_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prepare(args.data, args.coarse_predictions, args.output, args.split_manifest, args.split_seed)


if __name__ == "__main__":
    main()
