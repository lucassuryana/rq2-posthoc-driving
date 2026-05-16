"""
scripts/02_preprocess_target.py

Preprocess target samples: extract 7 CAM_FRONT frames (tm3..t3) and compute
past/future trajectories in vehicle-local coordinates.

Reads:  results/selected_samples_target.json
Writes: results/preprocessed_target.json
        cache/frames/<sample_token>/{tm3,tm2,tm1,t0,t1,t2,t3}.jpg
"""
import json, os, math
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# ── Config ────────────────────────────────────────────────────────────────────
NUSC_VERSION = 'v1.0-trainval'
NUSC_ROOT    = '/tudelft.net/staff-umbrella/lsuryana/rq2-posthoc-driving/data/nuscenes'
INPUT_PATH   = 'results/selected_samples_target.json'
OUTPUT_PATH  = 'results/preprocessed_target.json'

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading nuScenes {NUSC_VERSION}...")
nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_ROOT, verbose=False)

with open(INPUT_PATH) as f:
    selected = json.load(f)
print(f"Preprocessing {len(selected)} samples")


def get_ego_pose(nusc, sample_token):
    sample   = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    return np.array(ego_pose['translation']), Quaternion(ego_pose['rotation'])


def global_to_vehicle(pos_global, ref_pos, ref_rot):
    delta = pos_global[:2] - ref_pos[:2]
    yaw   = ref_rot.yaw_pitch_roll[0]
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    x_veh =  cos_y * delta[0] - sin_y * delta[1]
    y_veh =  sin_y * delta[0] + cos_y * delta[1]
    return float(x_veh), float(y_veh)


def save_frame(sample_token, dst_path):
    if os.path.exists(dst_path):
        return
    sample = nusc.get('sample', sample_token)
    cam    = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    src    = os.path.join(NUSC_ROOT, cam['filename'])
    Image.open(src).convert('RGB').save(dst_path, quality=90)


preprocessed = []

for idx, entry in enumerate(selected):
    sample_token  = entry['sample_token']
    future_tokens = entry['future_tokens']
    frame_dir     = f"cache/frames/{sample_token}"
    os.makedirs(frame_dir, exist_ok=True)

    ref_pos, ref_rot = get_ego_pose(nusc, sample_token)

    # ── Future trajectory (t+1..t+3) ──
    trajectory = []
    for tok in future_tokens:
        pos, _ = get_ego_pose(nusc, tok)
        x, y   = global_to_vehicle(pos, ref_pos, ref_rot)
        trajectory.append({'x': round(x, 3), 'y': round(y, 3)})

    # ── t0 ──
    save_frame(sample_token, f"{frame_dir}/t0.jpg")

    # ── Future frames (t+1..t+3) ──
    for i, tok in enumerate(future_tokens, start=1):
        save_frame(tok, f"{frame_dir}/t{i}.jpg")

    # ── Past frames (t-1..t-3) + past trajectory ──
    past_tokens = []
    tok = nusc.get('sample', sample_token)['prev']
    for i in range(1, 4):
        if tok == '':
            break
        past_tokens.append(tok)
        save_frame(tok, f"{frame_dir}/tm{i}.jpg")
        tok = nusc.get('sample', tok)['prev']

    n_past = len(past_tokens)
    past_trajectory = []
    for tok in reversed(past_tokens):
        pos, _ = get_ego_pose(nusc, tok)
        x, y   = global_to_vehicle(pos, ref_pos, ref_rot)
        past_trajectory.append({'x': round(x, 3), 'y': round(y, 3)})

    preprocessed.append({
        **entry,
        'frame_dir':       frame_dir,
        'trajectory':      trajectory,
        'n_past':          n_past,
        'past_trajectory': past_trajectory,
    })
    print(f"[{idx+1}/{len(selected)}] {entry.get('scene_name','?')} "
          f"{sample_token[:8]}... n_past={n_past}")

with open(OUTPUT_PATH, 'w') as f:
    json.dump(preprocessed, f, indent=2)
print(f"\nDone. Saved {len(preprocessed)} samples to {OUTPUT_PATH}")
