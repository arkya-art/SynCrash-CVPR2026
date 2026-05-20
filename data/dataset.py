import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

# VideoMAE standard normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

METADATA_VOCABS = {
    "scene_layout": ["highway", "signalized_intersection", "simple_intersection", 
                     "grade_separated_intersection", "city_street", "tunnel", "parking_lot", "UNKNOWN"],
    "weather": ["normal", "rain", "snow", "UNKNOWN"],
    "day_time": ["day", "night", "UNKNOWN"]
}

def apply_heavy_degradation_clip(frames_np: np.ndarray, is_validation: bool = False) -> np.ndarray:
    """
    Applies augmentations to a 16-frame [16, 224, 224, 3] clip.
    Frames are expected to be uint8.
    """
    aug_frames = frames_np.copy()
    T, H, W, C = aug_frames.shape
    
    if is_validation or random.random() < 0.7:
        scale_factor = random.uniform(0.15, 0.5) if not is_validation else 0.25
        new_h, new_w = int(H * scale_factor), int(W * scale_factor)
        for i in range(T):
            small = cv2.resize(aug_frames[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            aug_frames[i] = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
            
    if is_validation or random.random() < 0.5:
        ksize = random.choice([5, 7, 9, 11]) if not is_validation else 9
        kernel = np.zeros((ksize, ksize))
        direction = random.choice(['h', 'v', 'd1', 'd2'])
        if direction == 'h': kernel[ksize//2, :] = 1
        elif direction == 'v': kernel[:, ksize//2] = 1
        else: np.fill_diagonal(kernel, 1)
        kernel /= ksize
        for i in range(T):
            aug_frames[i] = cv2.filter2D(aug_frames[i], -1, kernel)

    if is_validation or random.random() < 0.5:
        sigma = random.uniform(0.5, 2.0) if not is_validation else 1.5
        noise_level = random.uniform(5, 25) if not is_validation else 15
        for i in range(T):
            blurred = cv2.GaussianBlur(aug_frames[i], (5, 5), sigma)
            noise = np.random.randn(*blurred.shape) * noise_level
            aug_frames[i] = np.clip(blurred + noise, 0, 255).astype(np.uint8)

    if not is_validation and random.random() < 0.5:
        alpha = random.uniform(0.5, 1.5)
        beta = random.uniform(-30, 30)
        aug_frames = np.clip(aug_frames * alpha + beta, 0, 255).astype(np.uint8)

    if not is_validation and random.random() < 0.3:
        h_erase, w_erase = random.randint(20, H//3), random.randint(20, W//3)
        y, x = random.randint(0, H - h_erase), random.randint(0, W - w_erase)
        aug_frames[:, y:y+h_erase, x:x+w_erase, :] = 127

    return aug_frames


class BinaryClipDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, mode: str = 'train', 
                 apply_degradation: bool = False,
                 valid_video_names: list = None):
        """
        csv_path: path to binary_clips_metadata.csv
        root_dir: directory containing 'binary_clips' folder
        mode: 'train' or 'val'. Only used to decide strictness of augmentations.
        valid_video_names: optionally filter dataset to specific videos (for K-Fold)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.apply_degradation = apply_degradation
        
        df = pd.read_csv(csv_path)
        if valid_video_names is not None:
            df = df[df['video_name'].isin(valid_video_names)].reset_index(drop=True)
            
        self.samples = df.to_dict('records')
        
        # Calculate sample weights for the sampler
        labels = df['label'].values
        class_counts = np.bincount(labels) # e.g. [num_zeros, num_ones]
        
        if len(class_counts) == 2 and class_counts[0] > 0 and class_counts[1] > 0:
            # inverse frequency weights
            class_weights = 1.0 / class_counts
            self.sample_weights = np.array([class_weights[l] for l in labels])
        else:
            self.sample_weights = np.ones(len(labels))
            
    def _get_metadata_indices(self, sample):
        def _idx(v_name, val):
            vocab = METADATA_VOCABS[v_name]
            return vocab.index(val) if val in vocab else vocab.index("UNKNOWN")
        
        return {
            "scene_idx": _idx("scene_layout", sample.get("scene_layout", "UNKNOWN")),
            "weather_idx": _idx("weather", sample.get("weather", "UNKNOWN")),
            "day_time_idx": _idx("day_time", sample.get("day_time", "UNKNOWN")),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_rel_path = sample['clip_path']
        clip_abs_path = os.path.join(self.root_dir, clip_rel_path)
        
        label = torch.tensor(sample['label'], dtype=torch.float32)
        meta_indices = self._get_metadata_indices(sample)
        
        try:
            # Loaded as [16, 224, 224, 3], uint8 CPU tensor
            frames_tensor = torch.load(clip_abs_path, map_location='cpu', weights_only=False)
            frames_np = frames_tensor.numpy()
        except Exception as e:
            # Fallback zero tensor if file is broken
            print(f"Error loading {clip_abs_path}: {e}")
            frames_np = np.zeros((16, 224, 224, 3), dtype=np.uint8)
            label = torch.tensor(0.0)
            
        if self.mode == 'train' or self.apply_degradation:
            frames_np = apply_heavy_degradation_clip(frames_np, is_validation=(self.mode == 'val'))
            
        # Convert to float and reorder to [C, T, H, W] for VideoMAE
        frames_pt = torch.from_numpy(frames_np).float() / 255.0
        frames_pt = frames_pt.permute(3, 0, 1, 2) # [C, T, H, W]
        
        # Normalize
        frames_pt = (frames_pt - MEAN) / STD
        
        return {
            "pixel_values": frames_pt,
            "labels": label,
            "scene_idx": torch.tensor(meta_indices['scene_idx'], dtype=torch.long),
            "weather_idx": torch.tensor(meta_indices['weather_idx'], dtype=torch.long),
            "day_time_idx": torch.tensor(meta_indices['day_time_idx'], dtype=torch.long),
            "video_name": sample['video_name'],
            "start_time": sample['start_time'],
            "end_time": sample['end_time']
        }
