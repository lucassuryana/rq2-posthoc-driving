"""
scripts/04_drivelm_label_stats.py
Discover the empirical thresholds DriveLM annotators implicitly used.

For every DriveLM-labelled keyframe whose scene is in nuScenes mini:
  1. Extract DriveLM's direction + speed labels from the behavior QA.
  2. Compute future trajectory in t0 ego-frame using nuScenes ego poses.
  3. Compute avg_speed_mps, acceleration_mps2, angle_overall_deg, yaw_rate_deg_s.
Then group by (direction_label, speed_label) and report descriptive stats.

Output: config/drivelm_label_stats.json + console summary tables.
"""
import os, json, math, re
import numpy as np
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

NUSCENES_ROOT    = os.environ.get('NUSCENES_ROOT',
                                  os.path.expanduser('~/rq2_experiment/data/nuscenes'))
NUSCENES_VERSION = os.environ.get('NUSCENES_VERSION', 'v1.0-mini')
DRIVELM_DIR      = os.environ.get('DRIVELM_DIR',
                                  os.path.expanduser('~/rq2_experiment/data/drivelm'))
DRIVELM_FILES = ['v1_1_train_nus.json']   # drop val_v1_1.json until fixed
HORIZON_S        = 1.5
OUT_PATH         = 'config/drivelm_label_stats.json'

DIR_LABELS = ['steering to the left',
              'slightly steering to the left',
              'going straight',
              'slightly steering to the right',
              'steering to the right']
SPD_LABELS = ['not moving',
              'driving slowly',
              'driving with normal speed',
              'driving fast',
              'driving very fast']

# ── Load nuScenes mini ────────────────────────────────────────────────
print(f"Loading nuScenes {NUSCENES_VERSION}...")
nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=False)

def ego_pose_for(sample):
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    return nusc.get('ego_pose', sd['ego_pose_token'])

# Lookup: sample_token → sample
mini_samples = {s['token']: s for s in nusc.sample}
print(f"  {len(mini_samples)} sample tokens in mini.\n")

# ── Load DriveLM labels ───────────────────────────────────────────────
def parse_behavior_answer(text):
    """Extract direction + speed from a DriveLM behavior answer string."""
    direction = next((d for d in DIR_LABELS if d in text), None)
    speed     = next((s for s in SPD_LABELS if s in text), None)
    return direction, speed

records = []
labels_seen = 0
samples_in_mini = 0
for fn in DRIVELM_FILES:
    path = os.path.join(DRIVELM_DIR, fn)
    if not os.path.exists(path):
        print(f"  skip (not found): {fn}")
        continue
    print(f"Reading {fn}...")
    with open(path) as f:
        drivelm = json.load(f)
    for scene_token, scene in drivelm.items():
        for sample_token, kf in scene.get('key_frames', {}).items():
            qa = kf.get('QA', {})
            beh = qa.get('behavior', [])
            for q in beh:
                ans = q.get('A', '')
                d, s = parse_behavior_answer(ans)
                if d is None or s is None:
                    continue
                labels_seen += 1
                if sample_token not in mini_samples:
                    continue
                samples_in_mini += 1
                records.append({
                    'sample_token': sample_token,
                    'direction':    d,
                    'speed':        s,
                })
                break  # one behavior label per keyframe is enough

print(f"\nDriveLM labels parsed total: {labels_seen}")
print(f"Of those, in nuScenes mini:  {samples_in_mini}\n")

if not records:
    raise SystemExit("No overlapping samples — check DRIVELM_DIR and "
                     "ensure DriveLM JSON covers nuScenes mini scenes.")

# ── Compute physics fields per record ────────────────────────────────
def compute_physics(sample_token):
    sample = mini_samples[sample_token]
    cur, future, ok = sample, [], True
    for _ in range(3):
        if cur['next'] == '':
            return None
        cur = nusc.get('sample', cur['next'])
        future.append(cur)

    ref = ego_pose_for(sample)
    q_inv = Quaternion(ref['rotation']).inverse
    ref_t = np.array(ref['translation'])

    pts = [(0.0, 0.0)]
    for fut in future:
        ep = ego_pose_for(fut)
        local = q_inv.rotate(np.array(ep['translation']) - ref_t)
        pts.append((float(local[0]), float(local[1])))

    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
    d1 = math.hypot(x1-x0, y1-y0)
    d2 = math.hypot(x2-x1, y2-y1)
    d3 = math.hypot(x3-x2, y3-y2)

    speed_1 = d1 / 0.5
    speed_3 = d3 / 0.5
    avg_speed    = (d1 + d2 + d3) / HORIZON_S
    acceleration = (speed_3 - speed_1) / 1.0

    angle_1 = math.atan2(y1-y0, x1-x0) if d1 > 0.1 else 0.0
    angle_3 = math.atan2(y3-y2, x3-x2) if d3 > 0.1 else 0.0
    total_dist = math.hypot(x3, y3)
    angle_overall = 0.0 if total_dist < 0.1 else math.atan2(y3, x3)
    yaw_rate = (angle_3 - angle_1) / 1.0  # rad/s

    return {
        'avg_speed_mps':     avg_speed,
        'acceleration_mps2': acceleration,
        'angle_overall_deg': math.degrees(angle_overall),
        'yaw_rate_deg_s':    math.degrees(yaw_rate),
        'x3_m':              x3,
        'y3_m':              y3,
    }

print("Computing physics for each labelled sample...")
enriched = []
for r in records:
    phys = compute_physics(r['sample_token'])
    if phys is None:
        continue
    r.update(phys)
    enriched.append(r)
print(f"  {len(enriched)} records with full 3-step future.\n")

# ── Group + describe ─────────────────────────────────────────────────
def describe(values):
    a = np.array(values, dtype=float)
    return {
        'n':    int(a.size),
        'mean': round(float(a.mean()),  3),
        'std':  round(float(a.std()),   3),
        'min':  round(float(a.min()),   3),
        'p25':  round(float(np.percentile(a, 25)), 3),
        'p50':  round(float(np.percentile(a, 50)), 3),
        'p75':  round(float(np.percentile(a, 75)), 3),
        'max':  round(float(a.max()),   3),
    }

PHYS_KEYS = ['avg_speed_mps', 'acceleration_mps2',
             'angle_overall_deg', 'yaw_rate_deg_s', 'x3_m', 'y3_m']

def stats_by(label_key, label_order):
    out = {}
    groups = defaultdict(list)
    for r in enriched:
        groups[r[label_key]].append(r)
    for L in label_order:
        rs = groups.get(L, [])
        if not rs:
            out[L] = {'n': 0}
            continue
        out[L] = {pk: describe([r[pk] for r in rs]) for pk in PHYS_KEYS}
        out[L]['n'] = len(rs)
    return out

speed_stats     = stats_by('speed',     SPD_LABELS)
direction_stats = stats_by('direction', DIR_LABELS)

# ── Print tables ─────────────────────────────────────────────────────
def print_table(title, stats, label_order, focus_keys):
    print("="*92)
    print(title)
    print("="*92)
    header = f"{'category':<32}{'n':>4}  " + "  ".join(
        f"{k:>22}" for k in focus_keys)
    print(header)
    print("-"*len(header))
    for L in label_order:
        s = stats[L]
        n = s.get('n', 0)
        if n == 0:
            print(f"{L:<32}{0:>4}  (no samples)")
            continue
        cells = []
        for k in focus_keys:
            d = s[k]
            cells.append(f"{d['p25']:>6.2f}/{d['p50']:>6.2f}/{d['p75']:>6.2f}")
        print(f"{L:<32}{n:>4}  " + "  ".join(f"{c:>22}" for c in cells))
    print("(cells show p25 / p50 / p75)\n")

print_table("SPEED CATEGORY → physics distribution",
            speed_stats, SPD_LABELS,
            ['avg_speed_mps', 'acceleration_mps2', 'x3_m'])

print_table("DIRECTION CATEGORY → physics distribution",
            direction_stats, DIR_LABELS,
            ['angle_overall_deg', 'yaw_rate_deg_s', 'y3_m'])

# ── Empirical thresholds ─────────────────────────────────────────────
def boundary(stats, label_order, key):
    """Boundary between two adjacent classes = midpoint of upper-class p25
       and lower-class p75 (overlap-aware)."""
    bounds = []
    for i in range(len(label_order)-1):
        lo = stats[label_order[i]]
        hi = stats[label_order[i+1]]
        if lo.get('n',0) == 0 or hi.get('n',0) == 0:
            bounds.append(None); continue
        b = 0.5*(lo[key]['p75'] + hi[key]['p25'])
        bounds.append(round(b, 3))
    return bounds

print("="*92)
print("EMPIRICAL THRESHOLDS (midpoint between adjacent classes)")
print("="*92)
spd_b = boundary(speed_stats,     SPD_LABELS, 'avg_speed_mps')
dir_b = boundary(direction_stats, DIR_LABELS, 'angle_overall_deg')
print(f"avg_speed_mps boundaries between {SPD_LABELS}:")
print(f"  {spd_b}")
print(f"angle_overall_deg boundaries between {DIR_LABELS}:")
print(f"  {dir_b}")

# ── Save ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out = {
    'source':           f'DriveLM ∩ nuScenes {NUSCENES_VERSION}',
    'n_records':        len(enriched),
    'horizon_s':        HORIZON_S,
    'speed_stats':      speed_stats,
    'direction_stats':  direction_stats,
    'speed_boundaries_avg_speed_mps':     spd_b,
    'direction_boundaries_angle_deg':     dir_b,
    'speed_labels':     SPD_LABELS,
    'direction_labels': DIR_LABELS,
}
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT_PATH}")
