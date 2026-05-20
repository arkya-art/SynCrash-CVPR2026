import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import torchvision.io as io
from torch.amp import autocast
from scipy.ndimage import gaussian_filter1d

from model_binary import BinaryAccidentModel
from dataset_binary_clips import MEAN, STD, METADATA_VOCABS

def get_spatial_center(unpooled_tokens, token_grads=None):
    """
    Extracts spatial center from VideoMAE unpooled tokens dynamically.
    unpooled_tokens: [SeqLen, Hidden]
    token_grads: [SeqLen, Hidden] (optional)
    
    Supported grids (tubelet_size=2):
      - (8, 16, 16) = 2048  <- VideoMAEv2-Giant, patch_size=14, 16-frame clip
      - (8, 14, 14) = 1568  <- VideoMAEv2-Base/Large, patch_size=16, 16-frame clip
      - (4, 16, 16) = 1024  <- Giant, 8-frame clip
      - (4, 14, 14) =  784  <- Base/Large, 8-frame clip
    """
    if token_grads is not None:
        # Grad-CAM: Global average pooling over the token gradients
        weights = token_grads.mean(dim=0)  # [Hidden]
        # Weighted combination of tokens
        saliency = (unpooled_tokens * weights).sum(dim=-1)  # [N]
        # ReLU to keep only features that have a positive influence on the "accident" class
        token_mags = torch.relu(saliency)
    else:
        # Fallback to feature magnitude
        token_mags = torch.norm(unpooled_tokens, dim=-1)  # [N]
        
    N = token_mags.shape[0]

    # Try candidate grids in priority order
    candidate_grids = [
        (8, 16, 16),   # 2048 - Giant w/ 16-frame clip
        (8, 14, 14),   # 1568 - Base/Large w/ 16-frame clip
        (4, 16, 16),   # 1024 - Giant w/ 8-frame clip
        (4, 14, 14),   #  784 - Base/Large w/ 8-frame clip
    ]

    for (T_t, H_t, W_t) in candidate_grids:
        if T_t * H_t * W_t == N:
            grid = token_mags.view(T_t, H_t, W_t)
            spatial_heatmap = grid.mean(dim=0)          # [H, W]
            max_idx = torch.argmax(spatial_heatmap)
            y, x = divmod(max_idx.item(), W_t)
            center_x = (x + 0.5) / W_t
            center_y = (y + 0.5) / H_t
            print(f"  Grid resolved: ({T_t},{H_t},{W_t}), peak patch=({x},{y})")
            return round(center_x, 3), round(center_y, 3)

    print(f"Warning: Unknown token count {N}. Defaulting to 0.5 center.")
    return 0.5, 0.5


def decode_video(path):
    try:
        vframes, _, info = io.read_video(path, pts_unit='sec', output_format='THWC')
        return vframes.numpy(), info.get('video_fps', 20.0)
    except Exception as e:
        print(f"Failed to decode {path}: {e}")
        return None, 20.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to a single test video")
    parser.add_argument("--model_weights", type=str, default="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/checkpoints/best_binary_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading Model from {args.model_weights}...")
    model = BinaryAccidentModel(freeze_backbone=False)
    state_dict = torch.load(args.model_weights, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 2. Decode Video
    print(f"Decoding video {args.video_path}...")
    frames, fps = decode_video(args.video_path)
    if frames is None:
        return

    T, H, W, C = frames.shape
    clip_len = 16
    stride = 2
    
    if T < clip_len:
        pad = np.repeat(frames[-1:], clip_len - T, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
        T = frames.shape[0]

    # Extract clips
    clips = []
    end_times = []
    for start in range(0, T - clip_len + 1, stride):
        end = start + clip_len
        clips.append(frames[start:end])
        end_times.append(end / fps)

    # 3. Dense Temporal Inference (Find the exact crash time)
    print(f"Running temporal inference across {len(clips)} clips...")
    
    # Dummy metadata (we don't strictly need precise metadata for spatial attention)
    s_t = torch.tensor([0], dtype=torch.long).to(device)
    w_t = torch.tensor([0], dtype=torch.long).to(device)
    d_t = torch.tensor([0], dtype=torch.long).to(device)
    
    probs = []
    MAX_BATCH = 8
    with torch.no_grad():
        for i in range(0, len(clips), MAX_BATCH):
            batch_clips = clips[i:i+MAX_BATCH]
            B = len(batch_clips)
            
            batch_tensors = []
            for clip in batch_clips:
                resized = np.zeros((clip_len, 224, 224, 3), dtype=np.uint8)
                for j in range(clip_len):
                    resized[j] = cv2.resize(clip[j], (224, 224), interpolation=cv2.INTER_LINEAR)
                t = torch.from_numpy(resized).float() / 255.0
                t = t.permute(3, 0, 1, 2)
                t = (t - MEAN) / STD
                batch_tensors.append(t)
                
            batch_t = torch.stack(batch_tensors).to(device)
            s_ext = s_t.expand(B).to(device)
            w_ext = w_t.expand(B).to(device)
            d_ext = d_t.expand(B).to(device)

            with autocast('cuda'):
                logits = model(batch_t, s_ext, w_ext, d_ext)
            
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            if B == 1: probs.append(batch_probs.item())
            else: probs.extend(batch_probs.tolist())

    # Detect Peak Time
    probs = np.array(probs)
    smoothed = gaussian_filter1d(probs, sigma=2.0)
    peak_idx = np.argmax(smoothed)
    accident_time = end_times[peak_idx]
    
    print(f"Temporal Peak Detected at: {accident_time:.2f}s (Probability: {smoothed[peak_idx]:.3f})")

    # 4. Spatial Localization (Zero-shot via forward hook on last transformer block)
    print(f"\nExtracting Spatial Heatmap for the Peak Clip...")
    peak_clip = clips[peak_idx]
    
    resized = np.zeros((clip_len, 224, 224, 3), dtype=np.uint8)
    for j in range(clip_len):
        resized[j] = cv2.resize(peak_clip[j], (224, 224), interpolation=cv2.INTER_LINEAR)
        
    t = torch.from_numpy(resized).float() / 255.0
    t = t.permute(3, 0, 1, 2)
    t = (t - MEAN) / STD
    t = t.unsqueeze(0).to(device) # [1, C, T, H, W]

    # Use a forward hook to capture the UNPOOLED output of the last transformer block
    # Also capture gradients on that output
    captured = {}
    
    def forward_hook(module, input, output):
        captured['tokens'] = output
        def backward_hook(grad):
            captured['grads'] = grad
        output.register_hook(backward_hook)
    
    # Register hook on the last transformer block
    last_block = model.backbone.model.blocks[-1]
    hook_handle = last_block.register_forward_hook(forward_hook)
    
    # We must enable gradients for the input to do a backward pass
    t.requires_grad_(True)
    
    model.eval()  # Keep model in eval mode (e.g. no dropout)
    
    with torch.set_grad_enabled(True):
        with autocast('cuda'):
            # Forward pass through the entire model to get the final logit
            s_ext = s_t.expand(1).to(device)
            w_ext = w_t.expand(1).to(device)
            d_ext = d_t.expand(1).to(device)
            
            logits = model(t, s_ext, w_ext, d_ext)
            target_score = logits[0]  # This is the "accident" logit
            
        # Backward pass from the logit to get gradients at the last transformer block
        model.zero_grad()
        target_score.backward()
    
    hook_handle.remove()  # Clean up the hook
    
    if 'tokens' in captured and 'grads' in captured:
        tokens = captured['tokens'].detach()  # [1, 2048, 1408]
        grads = captured['grads'].detach()    # [1, 2048, 1408]
        print(f"Captured unpooled tokens shape: {tokens.shape}")
        
        unpooled_tokens = tokens[0]  # [2048, 1408]
        token_grads = grads[0]       # [2048, 1408]
    else:
        print("Warning: Failed to capture tokens/gradients. Defaulting to 0.5.")
        unpooled_tokens = torch.zeros(2048, 1408)
        token_grads = torch.zeros(2048, 1408)

    # Calculate exact x, y center from the unpooled tokens and their gradients
    center_x, center_y = get_spatial_center(unpooled_tokens, token_grads)
    
    # Final Output
    print(f"\n{'-'*50}")
    print(f"✅ FINAL SUBMISSION OUTPUT:")
    print(f"path: {args.video_path}")
    print(f"accident_time: {round(accident_time, 2)}")
    print(f"center_x: {center_x}")
    print(f"center_y: {center_y}")
    print(f"type: single")
    print(f"{'-'*50}")

if __name__ == "__main__":
    main()
