import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.io as io
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def decode_video(path):
    try:
        vframes, _, info = io.read_video(path, pts_unit='sec', output_format='THWC')
        return vframes, info.get('video_fps', 20.0)
    except Exception as e:
        print(f"Failed to decode {path}: {e}")
        return None, 20.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to a single test video")
    parser.add_argument("--csv_path", type=str, default="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/submission_v4.csv")
    parser.add_argument("--prompt", type=str, default="crashed car. vehicle accident. collision.")
    parser.add_argument("--time_window", type=float, default=2.0)
    args = parser.parse_args()

    # 1. Load CSV to get the predicted accident time
    df = pd.read_csv(args.csv_path)
    # Find matching video row. Ensure exact match on basename just in case paths differ
    basename = os.path.basename(args.video_path)
    row = df[df['path'].str.contains(basename)]
    
    if len(row) == 0:
        print(f"Error: {basename} not found in {args.csv_path}")
        return
        
    accident_time = row.iloc[0]['accident_time']
    print(f"Found Temporal Prediction for {basename}: {accident_time:.2f}s")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Grounding DINO
    print("Loading GroundingDINO model...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    # 3. Decode Video
    print(f"Decoding video {args.video_path}...")
    vframes, fps = decode_video(args.video_path)
    if vframes is None: return
    
    # 4. Extract frames in the exact window: [accident_time - 2s, accident_time + 2s]
    start_time = max(0, accident_time - args.time_window)
    end_time = min(len(vframes)/fps, accident_time + args.time_window)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # We sample 10 frames evenly spaced in this window
    frame_indices = np.linspace(start_frame, end_frame, num=10, dtype=int)
    
    # Ensure they don't exceed the total frame count
    frame_indices = np.clip(frame_indices, 0, len(vframes)-1)
    
    print(f"Extracting {len(frame_indices)} frames around the peak time...")
    
    all_detections = []

    # 5. Run inference on each frame
    with torch.no_grad():
        for i, f_idx in enumerate(frame_indices):
            # vframes is THWC [T, H, W, C]
            img_tensor = vframes[f_idx]
            img_np = img_tensor.numpy()
            
            # Convert to PIL Image
            image = Image.fromarray(img_np)
            W, H = image.size
            
            # Grounding DINO requires lowercase prompt
            inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
            outputs = model(**inputs)

            # process outputs
            # set threshold to a reasonable value for zero-shot
            results = processor.image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([image.size[::-1]]), # (H, W)
                threshold=0.1
            )[0]
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score.item() > 0.1: # Low threshold internally to collect candidates
                    # box is [xmin, ymin, xmax, ymax]
                    xmin, ymin, xmax, ymax = box.tolist()
                    
                    # Convert to normalized center_x, center_y [0, 1]
                    center_x = ((xmin + xmax) / 2) / W
                    center_y = ((ymin + ymax) / 2) / H
                    
                    all_detections.append({
                        'frame_sec': f_idx / fps,
                        'score': score.item(),
                        'center_x': center_x,
                        'center_y': center_y,
                        'box': [round(b, 2) for b in [xmin, ymin, xmax, ymax]]
                    })
                    
    # 6. Aggregate Detections
    if len(all_detections) == 0:
        print("\nNo confident detections found in the window. Defaulting to 0.5, 0.5.")
        final_x = 0.5
        final_y = 0.5
    else:
        # Sort by highest score
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        best_det = all_detections[0]
        final_x = best_det['center_x']
        final_y = best_det['center_y']
        
        print("\nTop 3 Detections in window:")
        for det in all_detections[:3]:
            print(f"  Time: {det['frame_sec']:.2f}s | Score: {det['score']:.3f} | Center: ({det['center_x']:.3f}, {det['center_y']:.3f})")

    # 7. Final Output
    print(f"\n{'-'*50}")
    print(f"✅ FINAL SUBMISSION OUTPUT:")
    print(f"path: {args.video_path}")
    print(f"accident_time: {round(accident_time, 2)}")
    print(f"center_x: {round(final_x, 3)}")
    print(f"center_y: {round(final_y, 3)}")
    print(f"type: single")
    print(f"{'-'*50}")

if __name__ == "__main__":
    main()
