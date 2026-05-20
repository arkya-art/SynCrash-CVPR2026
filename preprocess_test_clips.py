"""
Stage 1: Preprocess test videos into sliding-window .pt clips (CPU-only).
This decodes each video, extracts overlapping 16-frame clips with stride=2,
resizes them to 224x224, and saves each clip as a .pt tensor.
A metadata CSV is produced that maps clip filenames -> video_name, end_time, metadata.

Run on a CPU-only SLURM node (no GPU needed):
    sbatch run_preprocess_test.sh
"""
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.io as io
from tqdm import tqdm

CLIP_LEN = 16
STRIDE = 2       # Dense stride, same as inference_binary.py
IMG_SIZE = 224

def decode_video(path):
    try:
        vframes, _, info = io.read_video(path, pts_unit='sec', output_format='THWC')
        fps = info.get('video_fps', 20.0)
        return vframes.numpy(), fps
    except Exception as e:
        print(f"Failed to decode {path}: {e}")
        return None, 20.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="/ihub/homedirs/sc_hrrs/arkya/test_metadata.csv")
    parser.add_argument("--video_dir", type=str, default="/ihub/homedirs/sc_hrrs/arkya")
    parser.add_argument("--output_dir", type=str, default="/ihub/homedirs/sc_hrrs/arkya/test_clips")
    parser.add_argument("--output_csv", type=str, default="/ihub/homedirs/sc_hrrs/arkya/test_clips/test_clips_metadata.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.test_csv)
    print(f"Loaded {len(df)} test videos from {args.test_csv}", flush=True)

    metadata_rows = []
    total_clips = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Test Videos"):
        video_rel = row['path']
        video_path = os.path.join(args.video_dir, video_rel)

        if not os.path.exists(video_path):
            print(f"Missing: {video_path}")
            continue

        frames, fps = decode_video(video_path)
        if frames is None or len(frames) == 0:
            continue

        T, H, W, C = frames.shape
        if T < CLIP_LEN:
            pad = np.repeat(frames[-1:], CLIP_LEN - T, axis=0)
            frames = np.concatenate([frames, pad], axis=0)
            T = frames.shape[0]

        # Make a safe basename for this video
        video_basename = os.path.splitext(os.path.basename(video_rel))[0]

        # Extract metadata from the row
        scene_layout = row.get('scene_layout', 'UNKNOWN')
        weather = row.get('weather', 'UNKNOWN')
        day_time = row.get('day_time', 'UNKNOWN')

        clip_idx = 0
        starts = list(range(0, T - CLIP_LEN + 1, STRIDE))
        # Make sure we always include the last possible clip
        if (T - CLIP_LEN) % STRIDE != 0 and T >= CLIP_LEN:
            last_start = T - CLIP_LEN
            if last_start not in starts:
                starts.append(last_start)

        for start in starts:
            end = start + CLIP_LEN
            clip_frames = frames[start:end]

            # Resize to 224x224
            resized = np.zeros((CLIP_LEN, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            for i in range(CLIP_LEN):
                resized[i] = cv2.resize(clip_frames[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

            clip_tensor = torch.from_numpy(resized)  # (16, 224, 224, 3) uint8
            out_filename = f"{video_basename}_c{clip_idx}.pt"
            torch.save(clip_tensor, os.path.join(args.output_dir, out_filename))

            metadata_rows.append({
                "clip_path": out_filename,
                "video_name": video_basename,
                "video_rel": video_rel,
                "end_time": end / fps,
                "scene_layout": scene_layout,
                "weather": weather,
                "day_time": day_time,
            })
            clip_idx += 1
            total_clips += 1

        if clip_idx > 0 and (total_clips % 500 == 0 or _ == len(df) - 1):
            print(f"  Progress: {total_clips} clips from {_ + 1} videos", flush=True)

    out_df = pd.DataFrame(metadata_rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\n{'='*60}", flush=True)
    print(f"Test preprocessing complete!", flush=True)
    print(f"  Total clips: {total_clips}", flush=True)
    print(f"  Total videos: {len(df)}", flush=True)
    print(f"  Output dir: {args.output_dir}", flush=True)
    print(f"  Metadata CSV: {args.output_csv}", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
