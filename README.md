# SynCrash: Multi-Stage Zero-Shot Accident Detection

**Accepted at CVPR 2026 (Workshop)** | Denver, Colorado  
**Authors:** Arkya Bagchi, Ritul Jangir, Varun Raskar (Autopilot, Indian Institute of Technology Jodhpur, India)  
**Contact:** [arkyabagchi1112@gmail.com](mailto:arkyabagchi1112@gmail.com) | [LinkedIn](https://www.linkedin.com/in/arkya-bagchi-11018461/)

---

## Overview
Accident detection in low-res CCTV is challenging due to compression and occlusions. 
Our key insight is that **temporal dynamics transfer better from synthetic data than spatial features.**

### Contributions
- **Modular three-stage pipeline (T, S, C).**
- **Domain-bridging VideoMAEv2 training.**
- **Physics-informed spatial localization.**

---

## Method

SynCrash decouples accident detection into three sequential stages:

### Stage 1: Temporal Localization
- Uses a **VideoMAEv2-giant backbone**.
- Employs **CCTV-degradation augmentations** (motion blur, compression artifacts, noise) and **metadata embeddings** to bridge the sim2real domain gap.

### Stage 2: Spatial Localization
- Leverages **YOLO detection + Trajectory estimation**.
- Uses a robust geometric priority cascade for impact localization:
  1. BBox Overlap Centroid
  2. Trajectory Intersection
  3. Size-weighted Midpoint

### Stage 3: Collision-Type Classification
- **Rule-based logic:** Classifies into head-on, rear-end, or t-bone based on the relative motion vectors of the detected vehicles.

---

## Quantitative Results

*Ranks 17th overall on the private leaderboard for the ACCIDENT@CVPR2026 benchmark.*

| Method | Public Score | Private Score |
| :--- | :---: | :---: |
| Graph-based interaction model | 0.27 | 0.25 |
| ViViT (joint multi-task) | 0.28 | 0.28 |
| RAFT-based motion modeling | 0.29 | 0.28 |
| VideoMAEv2 + Grad-CAM + rule-based | 0.34 | 0.33 |
| VideoMAEv2 + Q-former (query-based) | 0.37 | 0.36 |
| **VideoMAEv2 + YOLO + heuristic (Ours)** | **0.38** | **0.40** |

---

## Conclusions
- SynCrash achieves **0.38/0.40 (public/private)** on the benchmark.
- **Decoupling temporal and spatial reasoning is critical:** each task exhibits distinct sensitivity to domain shift.

### Limitations & Future Work
- Rule-based classification struggles with complex interactions.
- Future work will focus on improved interaction modeling and tighter stage integration with cross-stage feedback.

---

## Figures
See the root directory for high-resolution (`.png` and `.pdf`) diagrams used in the CVPR poster:
- `syncrash_pipeline` (Overall Architecture)
- `design_evolution` (Method Progression)
- `qualitative_localization` (Success/Failure Case Visualizations)
