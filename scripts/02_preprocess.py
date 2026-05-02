import json, os, math, shutil
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
with open('results/selected_samples.json') as f:
    selected = json.load(f)

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

preprocessed = []

for idx, entry in enumerate(selected):
    sample_token  = entry['sample_token']
    future_tokens = entry['future_tokens']
    frame_dir     = f"cache/frames/{sample_token}"
    os.makedirs(frame_dir, exist_ok=True)

    ref_pos, ref_rot = get_ego_pose(nusc, sample_token)

    # ── Future trajectory (t+1 to t+3) ───────────────────────────
    trajectory = []
    for tok in future_tokens:
        pos, _ = get_ego_pose(nusc, tok)
        x, y   = global_to_vehicle(pos, ref_pos, ref_rot)
        trajectory.append({'x': round(x, 3), 'y': round(y, 3)})

    # ── Current frame (t0) ───────────────────────────────────────
    sample  = nusc.get('sample', sample_token)
    cam     = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    src     = os.path.join('data/nuscenes', cam['filename'])
    dst     = f"{frame_dir}/t0.jpg"
    if not os.path.exists(dst):
        Image.open(src).convert('RGB').save(dst, quality=90)

    # ── Future frames (t+1 to t+3) ───────────────────────────────
    for i, tok in enumerate(future_tokens, start=1):
        cam = nusc.get('sample_data',
                  nusc.get('sample', tok)['data']['CAM_FRONT'])
        src = os.path.join('data/nuscenes', cam['filename'])
        dst = f"{frame_dir}/t{i}.jpg"
        if not os.path.exists(dst):
            Image.open(src).convert('RGB').save(dst, quality=90)

    # ── Past frames (t-1 to t-3) + past trajectory ───────────────
    past_tokens = []
    tok = nusc.get('sample', sample_token)['prev']
    for i in range(1, 4):
        if tok == '':
            break
        past_tokens.append(tok)
        cam = nusc.get('sample_data',
                  nusc.get('sample', tok)['data']['CAM_FRONT'])
        src = os.path.join('data/nuscenes', cam['filename'])
        dst = f"{frame_dir}/tm{i}.jpg"
        if not os.path.exists(dst):
            Image.open(src).convert('RGB').save(dst, quality=90)
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
    print(f"[{idx+1}/24] {sample_token[:8]}... "
          f"n_past={n_past} trajectory={trajectory}")

with open('results/preprocessed.json', 'w') as f:
    json.dump(preprocessed, f, indent=2)
print(f"\nDone. Saved {len(preprocessed)} samples to results/preprocessed.json")