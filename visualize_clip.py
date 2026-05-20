import os
import torch
import cv2
import sys

def visualize_clip(pt_path):
    clip = torch.load(pt_path) # [16, 224, 224, 3] uint8
    frames = clip.numpy()
    
    out_path = pt_path.replace(".pt", ".mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 5.0, (224, 224))
    
    for frame in frames:
        # Convert RGB to BGR for cv2
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)
        
    out.release()
    print(f"Saved visual clip to: {out_path}")

if __name__ == "__main__":
    visualize_clip(sys.argv[1])
