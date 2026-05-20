"""
Stage 2: Fast GPU-only inference on preprocessed test clips.
Loads pre-saved .pt clip tensors from disk (no video decoding!),
batches them through VideoMAEv2, and produces the final submission CSV.

Run on a GPU SLURM node:
    sbatch run_inference_fast.sh
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from model_binary import BinaryAccidentModel
from dataset_binary_clips import MEAN, STD, METADATA_VOCABS


class TestClipDataset(Dataset):
    """Loads preprocessed .pt clips and their metadata."""
    def __init__(self, clips_csv, clips_dir):
        self.df = pd.read_csv(clips_csv)
        self.clips_dir = clips_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_path = os.path.join(self.clips_dir, row['clip_path'])
        clip = torch.load(clip_path, weights_only=True)  # (16, 224, 224, 3) uint8

        # Normalize: convert to float, permute to (C, T, H, W), normalize
        t = clip.float() / 255.0
        t = t.permute(3, 0, 1, 2)  # (3, 16, 224, 224)
        t = (t - MEAN) / STD

        # Metadata indices
        s_raw = str(row.get('scene_layout', 'UNKNOWN'))
        w_raw = str(row.get('weather', 'UNKNOWN'))
        d_raw = str(row.get('day_time', 'UNKNOWN'))

        s_id = METADATA_VOCABS["scene_layout"].index(s_raw) if s_raw in METADATA_VOCABS["scene_layout"] else METADATA_VOCABS["scene_layout"].index("UNKNOWN")
        w_id = METADATA_VOCABS["weather"].index(w_raw) if w_raw in METADATA_VOCABS["weather"] else METADATA_VOCABS["weather"].index("UNKNOWN")
        d_id = METADATA_VOCABS["day_time"].index(d_raw) if d_raw in METADATA_VOCABS["day_time"] else METADATA_VOCABS["day_time"].index("UNKNOWN")

        return {
            "pixel_values": t,
            "scene_idx": s_id,
            "weather_idx": w_id,
            "day_time_idx": d_id,
            "video_name": row['video_name'],
            "video_rel": row['video_rel'],
            "end_time": row['end_time'],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_csv", type=str, default="/ihub/homedirs/sc_hrrs/arkya/test_clips/test_clips_metadata.csv")
    parser.add_argument("--clips_dir", type=str, default="/ihub/homedirs/sc_hrrs/arkya/test_clips")
    parser.add_argument("--model_weights", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="submission_v4_shriram.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BinaryAccidentModel(freeze_backbone=False)
    state_dict = torch.load(args.model_weights, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}", flush=True)

    # Dataset + DataLoader (fast parallel loading from disk)
    dataset = TestClipDataset(args.clips_csv, args.clips_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Loaded {len(dataset)} clips across test videos", flush=True)

    # Run batched inference
    all_video_names = []
    all_video_rels = []
    all_end_times = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="GPU Inference"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            scene_idx = batch["scene_idx"].to(device, non_blocking=True)
            weather_idx = batch["weather_idx"].to(device, non_blocking=True)
            day_time_idx = batch["day_time_idx"].to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(pixel_values, scene_idx, weather_idx, day_time_idx)

            probs = torch.sigmoid(logits).cpu().numpy()

            all_video_names.extend(batch["video_name"])
            all_video_rels.extend(batch["video_rel"])
            all_end_times.extend(batch["end_time"].numpy())
            all_probs.extend(probs)

    print(f"Inference done. Aggregating per-video predictions...", flush=True)

    # Group predictions by video
    clip_df = pd.DataFrame({
        "video_name": all_video_names,
        "video_rel": all_video_rels,
        "end_time": all_end_times,
        "prob": all_probs,
    })

    results = []
    for video_name, group in clip_df.groupby("video_name", sort=False):
        group = group.sort_values("end_time")
        probs = group["prob"].values
        end_times = group["end_time"].values
        video_rel = group["video_rel"].iloc[0]

        # Gaussian smoothing + peak detection
        smoothed = gaussian_filter1d(probs, sigma=2.0)
        peak_idx = np.argmax(smoothed)
        accident_time = end_times[peak_idx]

        print(f"  {video_rel}: pred_time={accident_time:.2f}s | prob_range=[{probs.min():.3f}, {probs.max():.3f}] | peak_prob={smoothed.max():.3f} | n_clips={len(probs)}", flush=True)

        results.append({
            "path": video_rel,
            "accident_time": round(accident_time, 2),
            "center_x": 0.5,
            "center_y": 0.5,
            "type": "rear-end",
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\n{'='*60}", flush=True)
    print(f"Submission saved to {args.out_csv}", flush=True)
    print(f"Total videos: {len(out_df)}", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
