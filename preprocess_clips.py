import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.io as io
import sys
from tqdm import tqdm

CLIP_LEN = 16
STRIDE = 8
FPS_ASSUMED = 20.0
IMG_SIZE = 224

def decode_video_frames(video_path: str) -> np.ndarray:
    try:
        # returns THWC, uint8
        vframes, _, _ = io.read_video(video_path, pts_unit='sec', output_format='THWC')
        return vframes.numpy()
    except Exception as e:
        print(f"Error decoding {video_path}: {e}")
        return None

def preprocess_and_save_clips():
    base_dir = "/ihub/homedirs/sc_hrrs/arkya"
    labels_path = os.path.join(base_dir, "sim_dataset/labels.csv")
    video_root = os.path.join(base_dir, "sim_dataset")
    output_dir = os.path.join(base_dir, "sim_dataset/binary_clips")
    output_csv = os.path.join(base_dir, "sim_dataset/binary_clips_metadata.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} videos from {labels_path}", flush=True)
    metadata_rows = []
    skipped_videos = 0
    total_clips_saved = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Clips"):
        rel_path = row.get("rgb_path", row.get("path"))
        video_path = os.path.join(video_root, rel_path)
        if not os.path.exists(video_path):
            print(f"File not found: {video_path}")
            continue
            
        accident_time = float(row.get("accident_time", -1.0))
        if accident_time < 0:
            continue
            
        duration = float(row.get("duration", 0))
        no_frames = int(row.get("no_frames", 0))
        fps = no_frames / duration if duration > 0 else FPS_ASSUMED
        
        # Map Carla weather to our shared vocab
        carla_weather = str(row.get("weather", "")).lower()
        weather = "normal"
        day_time = "day"
        if "rain" in carla_weather:
            weather = "rain"
        elif "snow" in carla_weather:
            weather = "snow"
            
        if "night" in carla_weather or "sunset" in carla_weather:
            day_time = "night"
            if carla_weather in ["night", "sunset"]:
                weather = "normal" # Defaulting if it was just 'night'

        scene_layout = "UNKNOWN"
        camera_position = row.get("camera_position", -1)
        
        frames = decode_video_frames(video_path)
        if frames is None or len(frames) == 0:
            continue
            
        T, H, W, C = frames.shape
        video_basename = os.path.splitext(os.path.basename(rel_path))[0]
        
        clip_idx = 0
        for start_idx in range(0, T - CLIP_LEN + 1, STRIDE):
            end_idx = start_idx + CLIP_LEN
            clip_frames = frames[start_idx:end_idx] # (16, H, W, 3)
            
            # Times
            start_time = start_idx / fps
            end_time = end_idx / fps
            
            # Label Assignment:
            # Positive: end_time in [accident_time - 1.5, accident_time + 0.5]
            # Negative: end_time < accident_time - 1.5
            # Discard: end_time > accident_time + 0.5 (post-crash)
            if accident_time - 1.5 <= end_time <= accident_time + 0.5:
                label = 1
            elif end_time < accident_time - 1.5:
                label = 0
            else:
                continue # Discard post-crash
            
            # Simple resize to 224x224 (uint8 storage)
            resized_clip = np.zeros((CLIP_LEN, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            for i in range(CLIP_LEN):
                resized_clip[i] = cv2.resize(clip_frames[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            
            clip_tensor = torch.from_numpy(resized_clip) # (16, 224, 224, 3)
            
            # Save format: Town05_sideswipe_rain_44_c0.pt
            # Replace slashes in basename if they exist e.g. videos/sideswipe/...
            safe_basename = video_basename.replace("/", "_")
            out_filename = f"{safe_basename}_c{clip_idx}.pt"
            out_filepath = os.path.join(output_dir, out_filename)
            torch.save(clip_tensor, out_filepath)
            
            metadata_rows.append({
                "clip_path": os.path.join("binary_clips", out_filename),
                "video_name": safe_basename,
                "label": label,
                "weather": weather,
                "day_time": day_time,
                "scene_layout": scene_layout,
                "start_time": start_time,
                "end_time": end_time
            })
            
            clip_idx += 1
            total_clips_saved += 1
        
        if clip_idx == 0:
            skipped_videos += 1
            
    out_df = pd.DataFrame(metadata_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"\n{'='*60}", flush=True)
    print(f"Extraction complete!", flush=True)
    print(f"  Total clips saved: {len(out_df)}", flush=True)
    print(f"  Skipped videos (0 clips): {skipped_videos}", flush=True)
    print(f"  Output dir: {output_dir}", flush=True)
    print(f"  Metadata CSV: {output_csv}", flush=True)
    print(f"\nLabel distribution:", flush=True)
    print(out_df['label'].value_counts().to_string(), flush=True)
    print(f"\nWeather distribution:", flush=True)
    print(out_df['weather'].value_counts().to_string(), flush=True)
    print(f"\nDay/Night distribution:", flush=True)
    print(out_df['day_time'].value_counts().to_string(), flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    preprocess_and_save_clips()
