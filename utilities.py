import hashlib
import json
import logging
import os
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils.data import Dataset


HISTORY_FRAMES = 16
FORECAST_FRAMES = 16
CLIP_FRAMES = HISTORY_FRAMES + FORECAST_FRAMES
PAPER_SPLIT_SEED = 42


def normalize_to_neg_one_to_one(tensor):
    """Map image tensors from [0, 1] to [-1, 1]."""
    return tensor * 2 - 1


def unnormalize_to_zero_to_one(tensor):
    """Map image tensors from [-1, 1] to [0, 1]."""
    return (tensor + 1) * 0.5


def move_batch_to_device(batch, device):
    """Move tensor values in a batch mapping while preserving metadata values."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def video_tensor_to_uint8(video):
    """Convert [...,C,H,W] video data in [0,1] to [...,H,W,C] uint8."""
    if not torch.is_tensor(video) or video.ndim < 3:
        raise ValueError("video must be a tensor with shape [...,C,H,W].")
    channel_last = torch.movedim(video.detach(), -3, -1)
    return (channel_last.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)


def load_dataset(hdf5_path=None):
    if hdf5_path is None:
        hdf5_path = os.path.join(os.path.dirname(__file__), "data", "video_frames.h5")
    with h5py.File(hdf5_path, "r") as f:
        required = {"start_idx", "imgs", "pv_output", "img_name"}
        missing = sorted(required.difference(f.keys()))
        if missing:
            raise KeyError(f"Dataset is missing HDF5 keys: {missing}")
        start_idx = f["start_idx"][...]
        sky_image = f["imgs"][...]
        pv_output = f["pv_output"][...]
        img_name = f["img_name"][...]
    img_name = [name.decode() if isinstance(name, bytes) else str(name) for name in img_name]
    return start_idx, sky_image, pv_output, img_name


def sample_spatial_transform(enabled: bool, probability: float):
    if not enabled or torch.rand(()) > probability:
        return 0, False, False
    return (
        int(torch.randint(0, 4, (1,)).item()),
        bool(torch.randint(0, 2, (1,)).item()),
        bool(torch.randint(0, 2, (1,)).item()),
    )


# Backward-compatible private alias for callers created before this helper was public.
_sample_spatial_transform = sample_spatial_transform


def apply_spatial_transform(clip, transform):
    """Apply one spatial transform to every frame and channel in a clip."""
    k, flip_vertical, flip_horizontal = transform
    if k:
        clip = torch.rot90(clip, k=k, dims=(-2, -1))
    if flip_vertical:
        clip = torch.flip(clip, dims=(-2,))
    if flip_horizontal:
        clip = torch.flip(clip, dims=(-1,))
    return clip


class SkyVideoDataset(Dataset):
    """Prepared 32-minute clips with explicit RGB/sun-mask fields."""

    def __init__(self, stride=1, hdf5_path=None, *, augment=False, aug_prob=0.5):
        if stride != 1:
            raise ValueError("The paper configuration requires one-minute stride=1.")
        if not 0.0 <= aug_prob <= 1.0:
            raise ValueError("aug_prob must be in [0, 1].")
        start_idx, sky_image, pv_output, img_name = load_dataset(hdf5_path)
        if sky_image.ndim != 4 or sky_image.shape[-1] != 4:
            raise ValueError("imgs must have shape [N,H,W,4] (RGB plus sun mask).")
        sky_image = rearrange(sky_image, "n h w c -> n c h w")
        if np.issubdtype(sky_image.dtype, np.integer):
            sky_image = sky_image.astype(np.float32) / 255.0
        else:
            sky_image = sky_image.astype(np.float32)
            if sky_image.max(initial=0.0) > 1.0:
                sky_image /= 255.0
        self.sky_image = torch.from_numpy(sky_image).contiguous().float()
        self.start_idx = np.asarray(start_idx, dtype=np.int64)
        if len(np.unique(self.start_idx)) != len(self.start_idx):
            raise ValueError("start_idx contains duplicates.")
        self.pv_output = torch.as_tensor(pv_output, dtype=torch.float32).reshape(-1, 1)
        self.img_name = img_name
        self.stride = stride
        self.augment = bool(augment)
        self.aug_prob = float(aug_prob)
        for stidx in self.start_idx:
            if stidx < 0 or stidx + CLIP_FRAMES > len(self.sky_image):
                raise IndexError(f"Clip start {stidx} does not contain {CLIP_FRAMES} frames.")

    def __len__(self):
        return len(self.start_idx)

    def get_item(self, idx, *, augment=None, transform=None):
        stidx = int(self.start_idx[idx])
        clip = self.sky_image[stidx : stidx + CLIP_FRAMES].clone()
        if transform is None:
            transform = sample_spatial_transform(
                self.augment if augment is None else augment, self.aug_prob
            )
        clip = apply_spatial_transform(clip, transform)
        pv = self.pv_output[stidx : stidx + CLIP_FRAMES]
        past, future = clip[:HISTORY_FRAMES], clip[HISTORY_FRAMES:]
        return {
            "start_idx": torch.tensor(stidx, dtype=torch.long),
            "past_rgb": past[:, :3],
            "past_sun": past[:, 3:4],
            "future_rgb": future[:, :3],
            "future_sun": future[:, 3:4],
            "past_pv": pv[:HISTORY_FRAMES, 0],
            "future_pv": pv[HISTORY_FRAMES:, 0],
        }, transform

    def __getitem__(self, idx):
        item, _ = self.get_item(idx)
        return item


def _load_predictions(prediction_path):
    with h5py.File(prediction_path, "r") as f:
        required = {"prediction_imgs", "eval_stidx"}
        missing = sorted(required.difference(f.keys()))
        if missing:
            raise KeyError(f"Prediction file is missing HDF5 keys: {missing}")
        frames = f["prediction_imgs"][...]
        indices = np.asarray(f["eval_stidx"][...], dtype=np.int64)
        metadata = {
            "split_manifest_hash": f.attrs.get("split_manifest_hash"),
            "architecture_version": f.attrs.get("architecture_version"),
        }
    if frames.ndim != 5 or frames.shape[1] != FORECAST_FRAMES or frames.shape[-1] != 3:
        raise ValueError("prediction_imgs must have shape [N,16,H,W,3].")
    if len(indices) != len(frames) or len(np.unique(indices)) != len(indices):
        raise ValueError("eval_stidx must be unique and aligned one-to-one with prediction_imgs.")
    frames = frames.astype(np.float32)
    if frames.max(initial=0.0) > 1.0:
        frames /= 255.0
    frames = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).contiguous()
    return {int(stidx): frames[pos] for pos, stidx in enumerate(indices)}, metadata


class HybridVideoDataset(Dataset):
    """Pair observations with PhyDiffNet predictions by start index."""

    def __init__(self, hdf5_path, prediction_path, *, augment=False, aug_prob=0.5):
        self.base = SkyVideoDataset(hdf5_path=hdf5_path, augment=False)
        self.predictions, self.prediction_metadata = _load_predictions(prediction_path)
        self.start_idx = self.base.start_idx
        self.img_name = self.base.img_name
        self.augment = bool(augment)
        self.aug_prob = float(aug_prob)
        expected = {int(value) for value in self.start_idx}
        provided = set(self.predictions)
        missing = expected - provided
        extra = provided - expected
        if missing or extra:
            raise ValueError(
                f"Prediction/base start_idx mismatch: {len(missing)} missing, {len(extra)} extra."
            )

    def __len__(self):
        return len(self.base)

    def get_item(self, idx, *, augment=None):
        enabled = self.augment if augment is None else augment
        transform = sample_spatial_transform(enabled, self.aug_prob)
        item, _ = self.base.get_item(idx, augment=False, transform=transform)
        stidx = int(item["start_idx"])
        item["future_generated_rgb"] = apply_spatial_transform(
            self.predictions[stidx].clone(), transform
        )
        return item

    def __getitem__(self, idx):
        return self.get_item(idx)


class DatasetView(Dataset):
    """A split view whose augmentation setting cannot leak to other splits."""

    def __init__(self, dataset, indices, *, augment=False):
        self.dataset = dataset
        self.indices = list(indices)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset.get_item(self.indices[idx], augment=self.augment)
        return item[0] if isinstance(item, tuple) else item


class BlurryVideoDataset(Dataset):
    """Aligned conditional/target clips for video-diffusion training."""

    def __init__(self, cond_vids, real_vids, *, augment=False, aug_prob=0.5):
        if cond_vids.shape[0] != real_vids.shape[0]:
            raise ValueError("Conditional and real videos must share the sample count.")
        if cond_vids.shape[1] != real_vids.shape[1]:
            raise ValueError("Conditional and real videos must share the channel count.")
        if cond_vids.shape[-2:] != real_vids.shape[-2:]:
            raise ValueError("Conditional and real videos must share spatial dimensions.")
        if not 0.0 <= aug_prob <= 1.0:
            raise ValueError("aug_prob must be in [0, 1].")
        self.cond_vids = cond_vids
        self.real_vids = real_vids
        self.augment = bool(augment)
        self.aug_prob = float(aug_prob)

    def __len__(self):
        return self.real_vids.shape[0]

    def __getitem__(self, idx):
        transform = sample_spatial_transform(self.augment, self.aug_prob)
        cond = apply_spatial_transform(self.cond_vids[idx].float(), transform)
        real = apply_spatial_transform(self.real_vids[idx].float(), transform)
        return cond, real


def load_video_dataset(hdf5_path):
    """Load a fixed 4-observed + 16-coarse conditional-diffusion dataset."""
    with h5py.File(hdf5_path, "r") as handle:
        required = {"real_vids", "pred_vids", "prev_vids"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"Diffusion dataset is missing HDF5 keys: {missing}")
        real_vids = handle["real_vids"][...]
        pred_vids = handle["pred_vids"][...]
        prev_vids = handle["prev_vids"][...]

    if real_vids.ndim != 5 or real_vids.shape[1] != FORECAST_FRAMES or real_vids.shape[-1] != 3:
        raise ValueError("real_vids must have shape [N,16,H,W,3].")
    if pred_vids.shape != real_vids.shape:
        raise ValueError("pred_vids must have the same shape as real_vids.")
    if prev_vids.ndim != 5 or prev_vids.shape[1] != 4 or prev_vids.shape[-1] != 3:
        raise ValueError("prev_vids must have shape [N,4,H,W,3].")

    def as_video_tensor(array):
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return normalize_to_neg_one_to_one(tensor)

    real_vids = as_video_tensor(real_vids)
    pred_vids = as_video_tensor(pred_vids)
    prev_vids = as_video_tensor(prev_vids)
    return torch.cat([prev_vids, pred_vids], dim=2), real_vids


def _date_for_clip(dataset, clip_index):
    stidx = int(dataset.start_idx[clip_index])
    return datetime.strptime(dataset.img_name[stidx], "%Y%m%d%H%M").date().isoformat()


def split_manifest_hash(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_split_manifest(manifest_path):
    """Load a persisted split manifest and return it with its stable hash."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return manifest, split_manifest_hash(manifest)


def validate_manifest_hash(actual, expected, artifact_name, *, allow_missing=False):
    """Reject artifacts produced from a different day-level data split."""
    if isinstance(actual, bytes):
        actual = actual.decode("utf-8")
    if actual in (None, ""):
        if allow_missing:
            return
        raise ValueError(f"{artifact_name} does not contain a split manifest hash.")
    if actual != expected:
        raise ValueError(f"{artifact_name} and requested split manifest hashes differ.")


def write_prediction_hdf5(path, frames, indices, *, manifest_hash,
                          architecture_version="paper-v1"):
    """Write aligned [N,16,3,H,W] predictions and their unique start indices."""
    if frames.ndim != 5 or frames.shape[1:3] != (FORECAST_FRAMES, 3):
        raise ValueError("frames must have shape [N,16,3,H,W].")
    indices = torch.as_tensor(indices, dtype=torch.long).detach().cpu().reshape(-1)
    if len(indices) != len(frames):
        raise ValueError("frames and indices must have the same sample count.")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("Prediction start indices must be unique.")
    order = torch.argsort(indices)
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(
            "prediction_imgs", data=video_tensor_to_uint8(frames[order]), compression="lzf"
        )
        handle.create_dataset("eval_stidx", data=indices[order].numpy(), compression="lzf")
        handle.attrs["architecture_version"] = architecture_version
        handle.attrs["split_manifest_hash"] = manifest_hash


def save_paper_checkpoint(path, model, optimizer, config, manifest, epoch, *,
                          architecture_version="paper-v1", normalization=None):
    """Save a strict, versioned checkpoint with split and normalization metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_values = dict(config) if isinstance(config, dict) else vars(config)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "architecture_version": architecture_version,
            "config": config_values,
            "split_manifest_hash": split_manifest_hash(manifest),
            "normalization": normalization or {"rgb": "[0,1]", "pv": "kW"},
        },
        output_path,
    )


def build_or_load_day_split(dataset, manifest_path=None, *, seed=PAPER_SPLIT_SEED):
    """Return clip indices for a deterministic 80/10/10 day-level split."""
    date_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        date_to_indices[_date_for_clip(dataset, idx)].append(idx)
    all_dates = sorted(date_to_indices)
    if len(all_dates) < 3:
        raise ValueError("At least three dates are required for train/validation/test splits.")
    path = Path(manifest_path) if manifest_path else None
    if path and path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("seed") != seed:
            raise ValueError("Existing split manifest seed does not match --split_seed.")
    else:
        rng = random.Random(seed)
        shuffled = all_dates.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(len(all_dates) * 0.1 + 0.5))
        n_val = max(1, int(len(all_dates) * 0.1 + 0.5))
        if n_test + n_val >= len(all_dates):
            raise ValueError("Not enough dates for the requested split.")
        manifest = {
            "version": 1,
            "seed": seed,
            "train": sorted(shuffled[n_test + n_val :]),
            "val": sorted(shuffled[n_test : n_test + n_val]),
            "test": sorted(shuffled[:n_test]),
        }
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assigned = manifest["train"] + manifest["val"] + manifest["test"]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(all_dates):
        raise ValueError("Split manifest dates do not match dataset dates exactly.")
    indices = {
        split: [idx for date in manifest[split] for idx in date_to_indices[date]]
        for split in ("train", "val", "test")
    }
    return indices, manifest


def split_dataset_by_date(dataset, manifest_path=None, *, seed=PAPER_SPLIT_SEED):
    indices, manifest = build_or_load_day_split(dataset, manifest_path, seed=seed)
    return (
        DatasetView(dataset, indices["train"], augment=True),
        DatasetView(dataset, indices["val"], augment=False),
        DatasetView(dataset, indices["test"], augment=False),
        manifest,
    )


def setup_logger(script_name, args):
    log_file_name = f"{script_name}_{datetime.now().strftime('%m%d-%H%M%S')}.log"
    log_folder = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_folder, log_file_name)), logging.StreamHandler()],
    )
    for arg, value in vars(args).items():
        logging.info("%s: %s", arg, value)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(req_mem=0):
    if not torch.cuda.is_available():
        logging.info("No GPU available. Using CPU.")
        return torch.device("cpu")
    torch.cuda.init()
    while True:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            check=True,
        )
        free_memory = [int(x) for x in result.stdout.decode("utf-8").strip().split("\n")]
        max_free_mem = max(free_memory)
        if max_free_mem >= req_mem:
            index = free_memory.index(max_free_mem)
            return torch.device(f"cuda:{index}")
        time.sleep(30)


def save_hyperparam(args):
    row = {"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **vars(args)}
    path = os.path.join(os.path.dirname(__file__), "logs", "hyperparam.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df = pd.concat([pd.read_csv(path), pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([row])
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df = df.sort_values("Time").drop_duplicates(subset=["model_name", "model_id"], keep="last")
    df.to_csv(path, index=False)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
