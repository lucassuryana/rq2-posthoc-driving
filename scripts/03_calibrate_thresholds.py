"""
scripts/03_calibrate_thresholds.py
Recalibrate DriveLM speed/direction thresholds using nuScenes train data.

Iterates every annotated keyframe with a 1.5s future horizon, transforms
future ego poses into the t0 ego frame, and reports thresholds via:
  - DriveLM-original applied to nuScenes (baseline / drift check)
  - Quantile-based recalibration

Output: config/thresholds_calibrated.json (drop-in for continuous_to_drivelm_label).

Run:
  NUSCENES_ROOT=/path/to/nuscenes python scripts/03_calibrate_thresholds.py
"""
import os, json, math
import numpy as np
from collections import Counter
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

NUSCENES_ROOT    = os.environ.get('NUSCENES_ROOT',    '/data/sets/nuscenes')
NUSCENES_VERSION = os.environ.get('NUSCENES_VERSION', 'v1.0-trainval')
SPLIT            = os.environ.get('NUSCENES_SPLIT',   'train')
HORIZON_S        = 1.5
OUT_PATH         = 'config/thresholds_calibrated.json'

# ── Load ─────────────────────────────────────────────────────────────
nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=False)
target_scenes = set(create_splits_scenes()[SPLIT])

def ego_pose_for(sample):
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    return nusc.get('ego_pose', sd['ego_pose_token'])

# ── Collect trajectories ─────────────────────────────────────────────
print(f"Reading {NUSCENES_VERSION} split='{SPLIT}'...")
records = []
for sample in nusc.sample:
    scene = nusc.get('scene', sample['scene_token'])
    if scene['name'] not in target_scenes:
        continue
    cur, future, ok = sample, [], True
    for _ in range(3):
        if cur['next'] == '':
            ok = False; break
        cur = nusc.get('sample', cur['next'])
        future.append(cur)
    if not ok:
        continue

    ref   = ego_pose_for(sample)
    q_inv = Quaternion(ref['rotation']).inverse
    ref_t = np.array(ref['translation'])

    traj = []
    for fut in future:
        ep = ego_pose_for(fut)
        local = q_inv.rotate(np.array(ep['translation']) - ref_t)
        traj.append((float(local[0]), float(local[1])))
    (x1, y1), (x2, y2), (x3, y3) = traj

    d1 = math.hypot(x1, y1)
    d2 = math.hypot(x2-x1, y2-y1)
    d3 = math.hypot(x3-x2, y3-y2)
    avg_speed = (d1 + d2 + d3) / HORIZON_S
    records.append({
        'avg_speed_mps': avg_speed,
        'x3': x3, 'y3': y3,
    })

n = len(records)
print(f"Collected {n} samples.\n")

speeds = np.array([r['avg_speed_mps'] for r in records])
x3s    = np.array([r['x3']            for r in records])
y3s    = np.array([r['y3']            for r in records])
abs_y  = np.abs(y3s)

# ── Distribution summary ─────────────────────────────────────────────
print("="*70)
print("DISTRIBUTION SUMMARY")
print("="*70)
print(f"avg_speed_mps   p10={np.percentile(speeds,10):.2f}  "
      f"p25={np.percentile(speeds,25):.2f}  p50={np.percentile(speeds,50):.2f}  "
      f"p75={np.percentile(speeds,75):.2f}  p90={np.percentile(speeds,90):.2f}")
print(f"x3 (forward m)  p10={np.percentile(x3s,10):.2f}  "
      f"p25={np.percentile(x3s,25):.2f}  p50={np.percentile(x3s,50):.2f}  "
      f"p75={np.percentile(x3s,75):.2f}  p90={np.percentile(x3s,90):.2f}")
print(f"|y3| (lateral)  p70={np.percentile(abs_y,70):.2f}  "
      f"p85={np.percentile(abs_y,85):.2f}  p90={np.percentile(abs_y,90):.2f}  "
      f"p95={np.percentile(abs_y,95):.2f}  p99={np.percentile(abs_y,99):.2f}")

# ── Helpers ──────────────────────────────────────────────────────────
SPEED_LABELS = ['not moving','driving slowly','driving with normal speed',
                'driving fast','driving very fast']
DIR_LABELS   = ['steering to the left','slightly steering to the left',
                'going straight',
                'slightly steering to the right','steering to the right']

def categorize_speed_x3(x, thresholds):
    for i, t in enumerate(thresholds):
        if x < t: return SPEED_LABELS[i]
    return SPEED_LABELS[-1]

def categorize_dir_y3(y, slight, full):
    if y >  full:   return DIR_LABELS[0]
    if y >  slight: return DIR_LABELS[1]
    if y < -full:   return DIR_LABELS[4]
    if y < -slight: return DIR_LABELS[3]
    return DIR_LABELS[2]

def show_dist(label, dist, n):
    print(f"\n  {label}")
    for L in dist:
        c = dist[L] if isinstance(dist, dict) else 0
    # Print in label order
    for L in (SPEED_LABELS if 'speed' in label.lower() else DIR_LABELS):
        c = dist.get(L, 0)
        print(f"    {L:35}: {c:5d} ({100*c/n:5.1f}%)")

# ── Baseline: DriveLM-original on nuScenes ──────────────────────────
print("\n" + "="*70)
print("BASELINE: DriveLM-original applied to nuScenes")
print("="*70)
ORIG_X3 = [2.0, 7.0, 10.0, 14.0]
ORIG_Y3_SLIGHT, ORIG_Y3_FULL = 0.5, 2.0
print(f"x3 thresholds: {ORIG_X3}")
print(f"y3 thresholds: ±{ORIG_Y3_SLIGHT}, ±{ORIG_Y3_FULL}")
spd0 = Counter(categorize_speed_x3(x, ORIG_X3) for x in x3s)
dir0 = Counter(categorize_dir_y3(y, ORIG_Y3_SLIGHT, ORIG_Y3_FULL) for y in y3s)
show_dist("speed distribution", dict(spd0), n)
show_dist("direction distribution", dict(dir0), n)

# ── Recalibration ────────────────────────────────────────────────────
print("\n" + "="*70)
print("RECALIBRATED (quantile-based)")
print("="*70)

# Speed: equal-frequency 5 bins on x3
x3_q = [float(np.percentile(x3s, p)) for p in [20, 40, 60, 80]]

# Direction: |y3| percentiles. p85 / p97 chosen so "going straight" remains
# the dominant class (~85%) and "full steering" stays rare (~3%) — preserves
# the spirit of the DriveLM categorisation without over-fragmenting.
y3_slight = float(np.percentile(abs_y, 85))
y3_full   = float(np.percentile(abs_y, 97))

print(f"x3 thresholds: {[round(t,2) for t in x3_q]}")
print(f"y3 thresholds: ±{y3_slight:.2f}, ±{y3_full:.2f}")
spd1 = Counter(categorize_speed_x3(x, x3_q) for x in x3s)
dir1 = Counter(categorize_dir_y3(y, y3_slight, y3_full) for y in y3s)
show_dist("speed distribution", dict(spd1), n)
show_dist("direction distribution", dict(dir1), n)

# ── Save ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out = {
    'method': 'quantile-x3 + percentile-y3',
    'source': f'nuScenes {NUSCENES_VERSION} split={SPLIT}',
    'n_samples': n,
    'horizon_s': HORIZON_S,
    'x3_thresholds_m': [round(t, 3) for t in x3_q],
    'y3_slight_m':     round(y3_slight, 3),
    'y3_full_m':       round(y3_full,   3),
    'speed_labels':     SPEED_LABELS,
    'direction_labels': DIR_LABELS,
}
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT_PATH}")
