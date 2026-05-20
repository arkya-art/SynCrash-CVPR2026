"""
hybrid_inference_2.py  (improved)
==================================
Hybrid strategy 2:
  - accident_time  : taken from submission_v5.csv
  - center_x/y     : improved spatial heuristic with:
      1. Overlap-region detection  – if boxes intersect, use overlap centroid
      2. Size-weighted midpoint    – larger objects pull the prediction toward them
      3. Trajectory extrapolation  – use recent frames to estimate where objects
                                     are heading and find their intersection point
      4. Approach-velocity ranking – pairs moving toward each other ranked higher

Strategy priority (highest → lowest):
  A. Overlap centroid of the intersecting pair with the largest combined area
  B. Trajectory-extrapolated intersection of the approaching pair
  C. Size-weighted midpoint of the closest approaching pair
  D. Closest-pair midpoint (original baseline)
  E. Single-object centre
  F. Frame centre fallback (0.5, 0.5)
"""
import os, sys, json, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
from config import BASE_DIR, TEST_ANN_DIR, TEST_META_CSV

# User-specified paths
BASELINE_V5_CSV = "/ihub/homedirs/sc_hrrs/arkya/accident_comp/submission_v5.csv"
OUTPUT_CSV      = os.path.join(BASE_DIR, "accident_comp/transformer_pipeline/submission_hybrid__exactframe_v2.csv")

# How many frames before the accident frame to use for trajectory estimation
TRAJECTORY_WINDOW = 5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_center(b):
    """Return (cx, cy) of a box [x1, y1, x2, y2]."""
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def bbox_area(b):
    """Return pixel area of a box."""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def size_weighted_midpoint(c1, a1, c2, a2):
    """
    Weighted average of two centers by their bounding-box areas.
    Larger object pulls the predicted collision point toward itself.
    """
    total = a1 + a2
    if total < 1e-6:
        return midpoint(c1, c2)
    wx = (c1[0] * a1 + c2[0] * a2) / total
    wy = (c1[1] * a1 + c2[1] * a2) / total
    return (wx, wy)

def overlap_centroid(b1, b2):
    """
    Return the centroid of the intersection rectangle, or None if they don't
    overlap.
    """
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 > ix1 and iy2 > iy1:
        return ((ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0)
    return None

def ray_intersection(p1, v1, p2, v2):
    """
    Find the closest point of approach between two rays:
      ray1: p1 + t*v1
      ray2: p2 + t*v2
    Returns the midpoint of the closest-approach segment, or None if rays
    are (nearly) parallel or moving apart.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # We solve the 2-ray closest-approach in 2-D using least-squares
    # | v1x  -v2x | |t|   |dx|
    # | v1y  -v2y | |s| = |dy|
    A = np.array([[v1[0], -v2[0]],
                  [v1[1], -v2[1]]], dtype=float)
    b = np.array([dx, dy], dtype=float)

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-6:          # parallel or stationary
        return None

    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    s = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det

    # Only accept forward-in-time intersections that are reasonably close
    if t < 0 or s < 0:
        return None

    hit1 = (p1[0] + t * v1[0], p1[1] + t * v1[1])
    hit2 = (p2[0] + s * v2[0], p2[1] + s * v2[1])
    return midpoint(hit1, hit2)


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def estimate_velocity(center_history):
    """
    Given a list of (cx, cy) positions across recent frames (oldest → newest),
    return an estimated velocity (vx, vy) via linear regression.
    Returns (0, 0) if fewer than 2 points are available.
    """
    if len(center_history) < 2:
        return (0.0, 0.0)
    xs = [p[0] for p in center_history]
    ys = [p[1] for p in center_history]
    ts = list(range(len(center_history)))
    vx = np.polyfit(ts, xs, 1)[0]
    vy = np.polyfit(ts, ys, 1)[0]
    return (float(vx), float(vy))

def build_track_histories(frames_with_boxes, bboxes_per_frame,
                          target_frame_idx, window):
    """
    For the `window` frames immediately before `target_frame_idx` in the
    frames_with_boxes list, collect per-object centre histories using a
    nearest-neighbour tracker (greedy IoU matching).

    Returns a list of dicts:
        {"center": (cx, cy), "velocity": (vx, vy), "box": [x1,y1,x2,y2],
         "area": float}
    one entry per object visible in the target frame, with its velocity
    estimated from prior frames.
    """
    if target_frame_idx >= len(frames_with_boxes):
        return []

    # Gather up to `window` frames ending AT target_frame_idx
    start = max(0, target_frame_idx - window)
    history_indices = list(range(start, target_frame_idx + 1))

    # Bootstrap tracks from the first window frame
    tracks = []   # each track is a list of (cx, cy)
    track_boxes = []

    for hi, fidx in enumerate(history_indices):
        current_boxes = bboxes_per_frame[fidx]
        current_centers = [bbox_center(b) for b in current_boxes]

        if hi == 0:
            tracks = [[c] for c in current_centers]
            track_boxes = [current_boxes[i] for i in range(len(current_boxes))]
            continue

        # Greedy nearest-neighbour matching
        prev_centers = [t[-1] for t in tracks]
        matched_track = [False] * len(tracks)
        new_tracks = []
        new_boxes  = []

        for ci, cc in enumerate(current_centers):
            if not prev_centers:
                new_tracks.append([cc])
                new_boxes.append(current_boxes[ci])
                continue
            dists = [euclidean(cc, pc) for pc in prev_centers]
            best  = int(np.argmin(dists))
            if not matched_track[best] and dists[best] < 200:   # px threshold
                tracks[best].append(cc)
                track_boxes[best] = current_boxes[ci]
                matched_track[best] = True
            else:
                new_tracks.append([cc])
                new_boxes.append(current_boxes[ci])

        # Add unmatched tracks (object disappeared — carry forward)
        for ti, tm in enumerate(matched_track):
            if not tm:
                tracks[ti].append(tracks[ti][-1])   # duplicate last position

        tracks     += new_tracks
        track_boxes += new_boxes

    # Build result for objects visible in the target frame
    result = []
    for ti, track in enumerate(tracks):
        if not track:
            continue
        vel = estimate_velocity(track)
        box = track_boxes[ti] if ti < len(track_boxes) else None
        if box is None:
            continue
        result.append({
            "center":   track[-1],
            "velocity": vel,
            "box":      box,
            "area":     bbox_area(box),
        })
    return result


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_center(incident_boxes, track_info, width, height):
    """
    Returns normalised (cx, cy) using the priority strategy described at the
    top of this file.

    Parameters
    ----------
    incident_boxes : list of [x1, y1, x2, y2]   — boxes at accident frame
    track_info     : list of dicts from build_track_histories (may be empty)
    width, height  : frame dimensions in pixels
    """

    def norm(pt):
        x = float(np.clip(pt[0] / width,  0.0, 1.0))
        y = float(np.clip(pt[1] / height, 0.0, 1.0))
        return x, y

    n = len(incident_boxes)

    # ── F: no objects ─────────────────────────────────────────────────────
    if n == 0:
        return 0.5, 0.5

    # ── E: single object ──────────────────────────────────────────────────
    if n == 1:
        return norm(bbox_center(incident_boxes[0]))

    # Build per-box metadata
    centers  = [bbox_center(b) for b in incident_boxes]
    areas    = [bbox_area(b)   for b in incident_boxes]

    # ── A: overlap region ─────────────────────────────────────────────────
    best_overlap = None
    best_overlap_score = -1.0
    for i, j in combinations(range(n), 2):
        oc = overlap_centroid(incident_boxes[i], incident_boxes[j])
        if oc is not None:
            score = areas[i] + areas[j]
            if score > best_overlap_score:
                best_overlap_score = score
                best_overlap = oc

    if best_overlap is not None:
        return norm(best_overlap)

    # Build velocity lookup from track_info (keyed by rounded center)
    vel_map = {}
    for t in track_info:
        key = (round(t["center"][0]), round(t["center"][1]))
        vel_map[key] = t["velocity"]

    def get_velocity(c):
        key = (round(c[0]), round(c[1]))
        return vel_map.get(key, (0.0, 0.0))

    # ── B: trajectory intersection ────────────────────────────────────────
    best_traj = None
    best_approach = float("inf")   # lower closing speed = lower priority

    for i, j in combinations(range(n), 2):
        ci, cj = centers[i], centers[j]
        vi = get_velocity(ci)
        vj = get_velocity(cj)

        # Closing speed: negative dot product of (relative velocity) · (separation)
        sep   = (cj[0] - ci[0], cj[1] - ci[1])
        rel_v = (vi[0] - vj[0],  vi[1] - vj[1])
        approach_speed = -(sep[0] * rel_v[0] + sep[1] * rel_v[1])

        if approach_speed <= 0:
            continue   # moving apart; skip

        hit = ray_intersection(ci, vi, cj, vj)
        if hit is not None and approach_speed > best_approach:
            # Favour pairs that are approaching fastest
            best_approach = approach_speed
            best_traj = hit

    if best_traj is not None:
        return norm(best_traj)

    # ── C: size-weighted midpoint of closest approaching pair ─────────────
    # Among pairs that are approaching each other (positive approach speed),
    # pick the closest one and apply size weighting.
    best_swm = None
    best_dist_approaching = float("inf")

    for i, j in combinations(range(n), 2):
        ci, cj = centers[i], centers[j]
        vi = get_velocity(ci)
        vj = get_velocity(cj)
        sep   = (cj[0] - ci[0], cj[1] - ci[1])
        rel_v = (vi[0] - vj[0],  vi[1] - vj[1])
        approach_speed = -(sep[0] * rel_v[0] + sep[1] * rel_v[1])
        dist = euclidean(ci, cj)
        if approach_speed > 0 and dist < best_dist_approaching:
            best_dist_approaching = dist
            best_swm = size_weighted_midpoint(ci, areas[i], cj, areas[j])

    if best_swm is not None:
        return norm(best_swm)

    # ── D: size-weighted midpoint of the overall closest pair ─────────────
    (bi, bj) = min(
        combinations(range(n), 2),
        key=lambda p: euclidean(centers[p[0]], centers[p[1]])
    )
    pt = size_weighted_midpoint(centers[bi], areas[bi], centers[bj], areas[bj])
    return norm(pt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[Hybrid-2] Loading baseline time from: {BASELINE_V5_CSV}")
    if not os.path.exists(BASELINE_V5_CSV):
        print(f"Error: {BASELINE_V5_CSV} not found.")
        return

    baseline_df = pd.read_csv(BASELINE_V5_CSV)
    metadata_df = pd.read_csv(TEST_META_CSV)
    meta_map    = {row["path"]: row for _, row in metadata_df.iterrows()}

    results = []
    # Counters for strategy usage
    strategy_counts = {"A_overlap": 0, "B_trajectory": 0,
                       "C_swm_approach": 0, "D_swm_closest": 0,
                       "E_single": 0, "F_fallback": 0}

    for _, brow in tqdm(baseline_df.iterrows(), total=len(baseline_df),
                        desc="Hybrid-2 Inference"):
        vid_path = brow["path"]
        acc_time = brow["accident_time"]

        cx, cy = 0.5, 0.5
        strategy = "F_fallback"
        acc_type = brow.get("type", "single")

        meta     = meta_map.get(vid_path, {})
        duration = float(meta.get("duration", 1.0))
        no_frames= float(meta.get("no_frames", 30))
        fps      = no_frames / max(duration, 1e-3)
        width    = float(meta.get("width", 1920))
        height   = float(meta.get("height", 1080))

        acc_frame = int(acc_time * fps)

        stem      = Path(vid_path).stem
        json_file = os.path.join(TEST_ANN_DIR, f"{stem}.json")

        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                info = json.load(f)

            width  = float(info.get("width",  width))
            height = float(info.get("height", height))

            frames_with_boxes = []
            bboxes_per_frame  = []

            for fi, frame_data in enumerate(info.get("frames_boxes", [])):
                f_boxes = []
                for obj in frame_data.get("objects", []):
                    try:
                        x1 = float(obj["2d_bbox"][0][0])
                        y1 = float(obj["2d_bbox"][0][1])
                        x2 = float(obj["2d_bbox"][1][0])
                        y2 = float(obj["2d_bbox"][1][1])
                        f_boxes.append([x1, y1, x2, y2])
                    except (KeyError, IndexError):
                        continue
                if f_boxes:
                    frames_with_boxes.append(fi)
                    bboxes_per_frame.append(f_boxes)

            if frames_with_boxes:
                available_frames = np.array(frames_with_boxes)
                idx_in_list = int((np.abs(available_frames - acc_frame)).argmin())
                incident_boxes  = bboxes_per_frame[idx_in_list]

                # Build trajectory information from preceding frames
                track_info = build_track_histories(
                    frames_with_boxes, bboxes_per_frame,
                    idx_in_list, TRAJECTORY_WINDOW
                )

                # ── Predict with improved heuristic ──────────────────────
                cx, cy = predict_center(incident_boxes, track_info,
                                        width, height)

                # Determine which strategy fired (for logging)
                n = len(incident_boxes)
                if n == 0:
                    strategy = "F_fallback"
                elif n == 1:
                    strategy = "E_single"
                    acc_type = "single"
                else:
                    acc_type = "t-bone"
                    centers = [bbox_center(b) for b in incident_boxes]
                    areas   = [bbox_area(b)   for b in incident_boxes]
                    # Re-check overlap
                    has_overlap = any(
                        overlap_centroid(incident_boxes[i], incident_boxes[j]) is not None
                        for i, j in combinations(range(n), 2)
                    )
                    if has_overlap:
                        strategy = "A_overlap"
                    else:
                        vel_map = {}
                        for t in track_info:
                            key = (round(t["center"][0]), round(t["center"][1]))
                            vel_map[key] = t["velocity"]

                        def get_v(c):
                            return vel_map.get((round(c[0]), round(c[1])), (0.0, 0.0))

                        has_traj = any(
                            -(
                                (centers[j][0]-centers[i][0])*(get_v(centers[i])[0]-get_v(centers[j])[0])
                               +(centers[j][1]-centers[i][1])*(get_v(centers[i])[1]-get_v(centers[j])[1])
                            ) > 0 and ray_intersection(centers[i], get_v(centers[i]),
                                                        centers[j], get_v(centers[j])) is not None
                            for i, j in combinations(range(n), 2)
                        )
                        if has_traj:
                            strategy = "B_trajectory"
                        else:
                            has_approach = any(
                                -(
                                    (centers[j][0]-centers[i][0])*(get_v(centers[i])[0]-get_v(centers[j])[0])
                                   +(centers[j][1]-centers[i][1])*(get_v(centers[i])[1]-get_v(centers[j])[1])
                                ) > 0
                                for i, j in combinations(range(n), 2)
                            )
                            strategy = "C_swm_approach" if has_approach else "D_swm_closest"

        strategy_counts[strategy] += 1
        results.append({
            "path":          vid_path,
            "accident_time": round(acc_time, 4),
            "center_x":      round(cx, 6),
            "center_y":      round(cy, 6),
            "type":          acc_type,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Done] {len(out_df)} predictions saved to {OUTPUT_CSV}")
    print("\nStrategy breakdown:")
    for k, v in strategy_counts.items():
        print(f"  {k:20s}: {v:5d}")


if __name__ == "__main__":
    main()