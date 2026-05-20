import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image

import sys
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import torchvision.io as io

def decode_video(path):
    """Decode the ENTIRE video — matches the single-video script exactly."""
    try:
        vframes, _, info = io.read_video(path, pts_unit='sec', output_format='THWC')
        return vframes, info.get('video_fps', 20.0)
    except Exception as e:
        print(f"Failed to decode {path}: {e}")
        return None, 20.0

def extract_frames_around_time(vframes, fps, target_time, time_window=2.0, num_frames=10):
    """Slice frames from pre-decoded video — matches the single-video script exactly."""
    start_time = max(0, target_time - time_window)
    end_time = min(len(vframes) / fps, target_time + time_window)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)
    frame_indices = np.clip(frame_indices, 0, len(vframes) - 1)

    frames = vframes[frame_indices].numpy()
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/submission_v4.csv")
    parser.add_argument("--out_csv", type=str, default="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/submission_dino.csv")
    parser.add_argument("--prompt", type=str, default="crashed car. vehicle accident. collision.")
    parser.add_argument("--time_window", type=float, default=2.0)
    parser.add_argument("--num_frames", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading GroundingDINO model...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    print(f"Starting batched inference on {len(df)} videos...")
    
    out_rows = []
    
    # We use tqdm for the progress bar, but also print the specific format to stdout
    pbar = tqdm(total=len(df), desc="Inference", file=sys.stderr)
    
    for idx, row in df.iterrows():
        video_path = row['path']
        # The CSV has 'videos/filename.mp4', we need absolute path to read
        # Base dir for videos is /ihub/homedirs/sc_hrrs/arkya
        abs_path = os.path.join("/ihub/homedirs/sc_hrrs/arkya", video_path)
        accident_time = row['accident_time']
        
        # Step 1: Decode the FULL video (matches single-video script)
        vframes, fps = decode_video(abs_path)
        
        final_x, final_y = 0.5, 0.5
        max_score = 0.0
        
        if vframes is not None and len(vframes) > 0:
            # Step 2: Extract frames around accident time from decoded video
            frames = extract_frames_around_time(vframes, fps, accident_time,
                                                time_window=args.time_window,
                                                num_frames=args.num_frames)
            del vframes  # Free memory immediately
            
            # Single-image inference to perfectly match the single video script
            for i, f in enumerate(frames):
                img_np = f
                image = Image.fromarray(img_np)
                W, H = image.size
                
                inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                results = processor.image_processor.post_process_object_detection(
                    outputs,
                    target_sizes=torch.tensor([image.size[::-1]]), # (H, W)
                    threshold=0.1
                )[0]
                
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    s = score.item()
                    if s > max_score and s > 0.1: # Only take if score > 0.1
                        max_score = s
                        xmin, ymin, xmax, ymax = box.tolist()
                        final_x = ((xmin + xmax) / 2) / W
                        final_y = ((ymin + ymax) / 2) / H
        
        # Format the output exactly like inference_v4_17950.log
        print(f"  {video_path}: pred_time={accident_time:.2f}s | dino_score={max_score:.3f} | center=({final_x:.3f}, {final_y:.3f})")

        out_rows.append({
            'path': row['path'],
            'accident_time': row['accident_time'],
            'center_x': round(final_x, 3),
            'center_y': round(final_y, 3),
            'type': row.get('type', 'single')
        })
        
        pbar.update(1)
        
        # Save incrementally every 50 rows
        if (idx + 1) % 50 == 0:
            pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
            
    pbar.close()
            
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\n✅ Finished! Saved final submission to {args.out_csv}")

if __name__ == "__main__":
    main()
