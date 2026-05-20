"""Quick diagnostic to isolate WHY dino_score=0.0 in batch mode."""
import os, sys, torch, numpy as np
from PIL import Image
import torchvision.io as io
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

VIDEO = "/ihub/homedirs/sc_hrrs/arkya/videos/__WFqm4i3vE_00.mp4"
PROMPT = "crashed car. vehicle accident. collision."
ACCIDENT_TIME = 7.48

print("="*60)
print("STEP 1: Decode video")
vframes, _, info = io.read_video(VIDEO, pts_unit='sec', output_format='THWC')
fps = info.get('video_fps', 20.0)
print(f"  vframes shape: {vframes.shape}, dtype: {vframes.dtype}, fps: {fps}")
print(f"  vframes min={vframes.min()}, max={vframes.max()}")

print("\nSTEP 2: Extract frames")
time_window = 2.0
start_time = max(0, ACCIDENT_TIME - time_window)
end_time = min(len(vframes) / fps, ACCIDENT_TIME + time_window)
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)
frame_indices = np.linspace(start_frame, end_frame, num=10, dtype=int)
frame_indices = np.clip(frame_indices, 0, len(vframes) - 1)
print(f"  Frame indices: {frame_indices}")
print(f"  start_frame={start_frame}, end_frame={end_frame}")

frames = vframes[frame_indices].numpy()
print(f"  frames shape: {frames.shape}, min={frames.min()}, max={frames.max()}")

print("\nSTEP 3: Load model")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
model.eval()

print("\nSTEP 4: Run inference on each frame")
for i, f in enumerate(frames):
    image = Image.fromarray(f)
    W, H = image.size
    print(f"\n  Frame {i}: size={W}x{H}")
    
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]),
        threshold=0.1
    )[0]
    
    print(f"    Num detections (>0.1): {len(results['scores'])}")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        s = score.item()
        xmin, ymin, xmax, ymax = box.tolist()
        cx = ((xmin + xmax) / 2) / W
        cy = ((ymin + ymax) / 2) / H
        print(f"    score={s:.4f} | center=({cx:.3f}, {cy:.3f}) | box=[{xmin:.1f},{ymin:.1f},{xmax:.1f},{ymax:.1f}]")

print("\n" + "="*60)
print("DONE")
