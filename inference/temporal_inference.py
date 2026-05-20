import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import torchvision.io as io
from torch.amp import autocast
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from models.videomae_accident import BinaryAccidentModel
from data.dataset import MEAN, STD, METADATA_VOCABS

def decode_video(path):
    try:
        vframes, _, info = io.read_video(path, pts_unit='sec', output_format='THWC')
        return vframes.numpy(), info.get('video_fps', 20.0)
    except Exception as e:
        print(f"Failed to decode {path}: {e}")
        return None, 20.0

def inference_video(model, video_path, metadata_row, device, clip_len=16, stride=2):
    frames, fps = decode_video(video_path)
    if frames is None or len(frames) == 0:
        return 0.0, [], [], []

    T, H, W, C = frames.shape
    if T < clip_len:
        pad = np.repeat(frames[-1:], clip_len - T, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
        T = frames.shape[0]

    clips = []
    end_times = []
    end_frames = []
    
    # Dense sliding window for inference
    for start in range(0, T - clip_len + 1, stride):
        end = start + clip_len
        clips.append(frames[start:end])
        end_times.append(end / fps)
        end_frames.append(end)

    if (T - clip_len) % stride != 0 and T >= clip_len:
        start = T - clip_len
        end = T
        if start not in [0] + list(range(stride, T - clip_len + 1, stride)):
            clips.append(frames[start:end])
            end_times.append(end / fps)
            end_frames.append(end)

    model.eval()
    probs = []

    # Map metadata to indices, gracefully handling missing
    s_raw = metadata_row.get('scene_layout', 'UNKNOWN')
    w_raw = metadata_row.get('weather', 'UNKNOWN')
    d_raw = metadata_row.get('day_time', 'UNKNOWN')
    
    s_id = METADATA_VOCABS["scene_layout"].index(s_raw) if s_raw in METADATA_VOCABS["scene_layout"] else METADATA_VOCABS["scene_layout"].index("UNKNOWN")
    w_id = METADATA_VOCABS["weather"].index(w_raw) if w_raw in METADATA_VOCABS["weather"] else METADATA_VOCABS["weather"].index("UNKNOWN")
    d_id = METADATA_VOCABS["day_time"].index(d_raw) if d_raw in METADATA_VOCABS["day_time"] else METADATA_VOCABS["day_time"].index("UNKNOWN")

    # We will expand these to match the batch size during inference
    scene_tensor = torch.tensor([s_id], dtype=torch.long)
    weather_tensor = torch.tensor([w_id], dtype=torch.long)
    day_tensor = torch.tensor([d_id], dtype=torch.long)

    MAX_BATCH = 8  # Process 8 overlapping clips concurrently on A100
    with torch.no_grad():
        for i in range(0, len(clips), MAX_BATCH):
            batch_clips = clips[i:i+MAX_BATCH]
            B = len(batch_clips)
            
            batch_tensors = []
            for clip in batch_clips:
                # Resize
                resized = np.zeros((clip_len, 224, 224, 3), dtype=np.uint8)
                for j in range(clip_len):
                    resized[j] = cv2.resize(clip[j], (224, 224), interpolation=cv2.INTER_LINEAR)
                
                t = torch.from_numpy(resized).float() / 255.0
                t = t.permute(3, 0, 1, 2)
                t = (t - MEAN) / STD
                batch_tensors.append(t)
                
            batch_t = torch.stack(batch_tensors).to(device, non_blocking=True) # [B, C, T, H, W]
            
            # Expand metadata tensors to match the dynamic batch size B
            s_t = scene_tensor.expand(B).to(device)
            w_t = weather_tensor.expand(B).to(device)
            d_t = day_tensor.expand(B).to(device)

            with autocast('cuda'):
                logits = model(batch_t, s_t, w_t, d_t)
            
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            
            if B == 1:
                probs.append(batch_probs.item())
            else:
                probs.extend(batch_probs.tolist())

    probs = np.array(probs)
    # 1D Gaussian smoothing to filter out isolated false positive spikes
    smoothed = gaussian_filter1d(probs, sigma=2.0)
    
    # Peak emergence: timestamp where accident probability is highest
    peak_idx = np.argmax(smoothed)
    accident_time = end_times[peak_idx]
    
    # The model predicts probability for a 16-frame clip.
    # The exact accident frame is typically the middle frame of this maximum probability clip.
    accident_frame = end_frames[peak_idx] - (clip_len // 2)
    
    return accident_time, accident_frame, probs, smoothed, end_times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="/ihub/homedirs/sc_hrrs/arkya/test_metadata.csv")
    parser.add_argument("--video_dir", type=str, default="/ihub/homedirs/sc_hrrs/arkya")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to trained best_binary_model.pth")
    parser.add_argument("--out_csv", type=str, default="submission_v4.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model without loading nexar weights for backbone, instead load our full fine-tuned weights
    model = BinaryAccidentModel(freeze_backbone=False)
    state_dict = torch.load(args.model_weights, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    df = pd.read_csv(args.test_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        video_rel = row['path']
        video_path = os.path.join(args.video_dir, video_rel)
        
        if not os.path.exists(video_path):
            print(f"Missing: {video_path}")
            continue
            
        acc_time, acc_frame, probs, smoothed, times = inference_video(model, video_path, row, device)
        
        if len(probs) > 0:
            print(f"  {video_rel}: pred_time={acc_time:.2f}s | pred_frame={acc_frame} | prob_range=[{min(probs):.3f}, {max(probs):.3f}] | peak_prob={max(smoothed):.3f} | n_clips={len(probs)}", flush=True)
        
        results.append({
            "path": video_rel,
            "accident_time": round(acc_time, 2),
            "accident_frame": acc_frame,
            "center_x": 0.5,
            "center_y": 0.5,
            "type": "rear-end"
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Inference complete! Saved predictions to {args.out_csv}")

if __name__ == "__main__":
    main()
