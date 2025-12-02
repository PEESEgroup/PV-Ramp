import os
import h5py
import numpy as np
import pandas as pd
import logging
import time
import random
import subprocess
from datetime import datetime
import torch
from torch.utils.data import Dataset, Subset
from einops import rearrange
from collections import defaultdict
from typing import Optional, Tuple

# loaddata
def load_dataset(hdf5_path=None):
    if hdf5_path is None:
        hdf5_path = os.path.join(os.path.dirname(__file__), "data", "video_frames.h5")
        
    with h5py.File(hdf5_path, 'r') as f:
        start_idx = f["start_idx"][...]
        sky_image = f["imgs"][...]
        pv_output = f["pv_output"][...]
        img_name = f["img_name"][...]

    img_name = [name.decode() for name in img_name]

    return start_idx, sky_image, pv_output, img_name
    
class SkyVideoDataset(Dataset):
    def __init__(
        self,
        stride: int = 1,
        hdf5_path: Optional[str] = None,
        *,
        augment: bool = False,
        aug_prob: float = 0.5
    ) -> None:
        if not (0.0 <= aug_prob <= 1.0):
            raise ValueError("aug_prob must be in [0, 1].")
        start_idx, sky_image, pv_output, img_name = load_dataset(hdf5_path)
        logging.info(f"Load sky images: {sky_image.shape}")

        # N H W C -> N C H W, float32 in [0,1]
        sky_image = rearrange(sky_image, "n h w c -> n c h w")
        sky_image = torch.from_numpy(sky_image / 255.0).contiguous().float()

        pv_output = torch.from_numpy(pv_output).float().unsqueeze(-1)

        self.sky_image = sky_image  # [N, C, H, W], N=time
        self.start_idx = start_idx  # list/array of clip starts
        self.pv_output = pv_output  # [N, 1]
        self.stride = int(stride)
        self.length = len(start_idx)
        self.img_name = img_name

        # Augmentation controls
        self.augment = bool(augment)
        self.aug_prob = float(aug_prob)

        logging.info(f"Dataset size: {self.length}. Augment={self.augment}, aug_prob={self.aug_prob}, stride={self.stride}")

    def __len__(self) -> int:
        return self.length

    @torch.no_grad()
    def _maybe_augment_clip(self, clip: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return clip
        if torch.rand(()) > self.aug_prob:
            return clip

        # Rotation k * 90 degrees
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            clip = torch.rot90(clip, k=k, dims=(-2, -1))

        # Optional flips
        if bool(torch.randint(0, 2, (1,)).item()):
            clip = torch.flip(clip, dims=(-2,))  # vertical (H)
        if bool(torch.randint(0, 2, (1,)).item()):
            clip = torch.flip(clip, dims=(-1,))  # horizontal (W)

        return clip

    def __getitem__(self, idx: int):
        stidx = int(self.start_idx[idx])

        image_data = self.sky_image[stidx : stidx + 32 : self.stride]
        image_data = self._maybe_augment_clip(image_data)

        input_len = image_data.shape[0] // 2
        input_frames = image_data[:input_len]
        target_frames = image_data[input_len:]

        pv_data = self.pv_output[stidx : stidx + 32 : self.stride]
        input_pv = pv_data[:input_len]
        target_pv = pv_data[input_len:]

        return [stidx, input_frames, target_frames, input_pv, target_pv]
    

def setup_logger(script_name, args):
    log_file_name = f"{script_name}_{datetime.now().strftime('%m%d-%H%M%S')}.log"
    log_folder = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_folder, exist_ok=True)

    log_file_path = os.path.join(log_folder, log_file_name)

    logging.basicConfig(
        level=logging.INFO,  # the lowest level of log records
        format='%(asctime)s - %(levelname)s - %(message)s',  # format
        handlers=[
            logging.FileHandler(log_file_path),  # output to file
            logging.StreamHandler()  # output to console
        ])

    for arg, value in vars(args).items(): 
        logging.info(f"{arg}: {value}")
            
    

def set_random_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(req_mem = 0):
    if torch.cuda.is_available():
        torch.cuda.init()
        while True:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE
            )
            free_memory = result.stdout.decode('utf-8').strip().split('\n')
            free_memory = [int(x) for x in free_memory]
            max_free_mem = max(free_memory)
            
            if max_free_mem >= req_mem:
                best_gpu_index = free_memory.index(max_free_mem)
                device = torch.device(f'cuda:{best_gpu_index}')
                logging.info(f"Found {len(free_memory)} GPU(s). Using GPU:{best_gpu_index} with free memory {max_free_mem}MB.")
                return device
            else:
                logging.info(f"Waiting for free memory to reach {req_mem}MB (Current max: {max_free_mem}MB). ")
                time.sleep(30)
    else:
        device = torch.device('cpu')
        logging.info("No GPU available. Using CPU.")
    return device





def save_hyperparam(args):
    df_row_dict = {'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    df_row_dict.update(vars(args))
    df_row = pd.DataFrame([df_row_dict])

    hyperparam_file = os.path.join(os.path.dirname(__file__), 'logs', 'hyperparam.csv')
    os.makedirs(os.path.dirname(hyperparam_file), exist_ok=True)

    try:
        df = pd.read_csv(hyperparam_file)
        df = pd.concat([df, df_row], ignore_index=True)
    except FileNotFoundError:
        df = df_row

    df['Time'] = pd.to_datetime(df['Time'], format = 'mixed')
    df = df.sort_values('Time', ascending=False)
    df = df.drop_duplicates(subset=['model_name', 'model_id'], keep='first')
    df = df.sort_values('Time', ascending=True)

    df.to_csv(hyperparam_file, index=False)
    logging.info(f"Hyperparameters saved to {hyperparam_file}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# setup dataset and dataloader
def split_dataset_by_date(dataset):
    date_to_idx = defaultdict(list)

    for idx, stidx in enumerate(dataset.start_idx):
        ts = dataset.img_name[stidx]
        date = datetime.strptime(ts, '%Y%m%d%H%M').date()
        date_to_idx[date].append(idx)

    all_dates = list(date_to_idx.keys())
    n = len(all_dates)
    if n == 0:
        raise ValueError("No dates parsed from dataset.")

    num_test_dates = max(1, int(n * 0.1))
    num_val_dates  = max(1, int(n * 0.1))
    if num_test_dates + num_val_dates >= n:
        num_val_dates = max(1, n - num_test_dates - 1)

    test_dates = set(random.sample(all_dates, num_test_dates))
    remaining_dates = [d for d in all_dates if d not in test_dates]
    val_dates = set(random.sample(remaining_dates, num_val_dates))

    test_idxs, val_idxs, train_idxs = [], [], []
    for date, idxs in date_to_idx.items():
        if date in test_dates:
            test_idxs.extend(idxs)
        elif date in val_dates:
            val_idxs.extend(idxs)
        else:
            train_idxs.extend(idxs)

    train_dataset = Subset(dataset, train_idxs)
    val_dataset   = Subset(dataset, val_idxs)
    test_dataset  = Subset(dataset, test_idxs)

    return train_dataset, val_dataset, test_dataset



