"""
scripts/01_select_target_scenes.py

Select DriveLM keyframes for a specific list of target scenes.
Uses nuScenes v1.0-trainval (full dataset, not mini) and DriveLM train split
(since most target scenes are in DriveLM train, not val).

Output: results/selected_samples_target.json
"""
import json
from nuscenes.nuscenes import NuScenes

# ── Config ────────────────────────────────────────────────────────────────────
NUSC_VERSION = 'v1.0-trainval'
NUSC_ROOT    = '/tudelft.net/staff-umbrella/lsuryana/rq2-posthoc-driving/data/nuscenes'
DRIVELM_PATH = '/home/nfs/lsuryana/rq2-posthoc-driving/data/drivelm/v1_1_train_nus.json'
OUTPUT_PATH  = 'results/selected_samples_target.json'

TARGET_SCENES = [
    'scene-0001', 'scene-0002', 'scene-0004', 'scene-0007',
    'scene-0011', 'scene-0019', 'scene-0024', 'scene-0025',
    'scene-0871', 'scene-1084', 'scene-1099', 'scene-1105',
]

# ── Load nuScenes + DriveLM ───────────────────────────────────────────────────
print(f"Loading nuScenes {NUSC_VERSION}...")
nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_ROOT, verbose=False)

print(f"Loading DriveLM annotations from {DRIVELM_PATH}...")
with open(DRIVELM_PATH) as f:
    dlm = json.load(f)

# Map nuScenes scene names ↔ scene tokens
scene_name_to_token = {s['name']: s['token'] for s in nusc.scene}
scene_token_to_name = {s['token']: s['name'] for s in nusc.scene}

# Map sample tokens → scene tokens for quick lookup
sample_to_scene = {s['token']: s['scene_token'] for s in nusc.sample}

# Validate target scenes exist
missing = [n for n in TARGET_SCENES if n not in scene_name_to_token]
if missing:
    print(f"WARNING: these scenes are not in nuScenes trainval: {missing}")

target_scene_tokens = {scene_name_to_token[n] for n in TARGET_SCENES
                       if n in scene_name_to_token}

# ── Select DriveLM keyframes from target scenes ──────────────────────────────
selected = []
per_scene_counts = {}

for drivelm_scene_id, scene_data in dlm.items():
    for sample_token, entry in scene_data['key_frames'].items():
        # Skip samples not in nuScenes (shouldn't happen for trainval)
        if sample_token not in sample_to_scene:
            continue
        # Only keep samples from target scenes
        scene_token = sample_to_scene[sample_token]
        if scene_token not in target_scene_tokens:
            continue

        qa = entry.get('QA', {})
        # Use planning question if available; else behavior; else perception
        question = None
        for category in ('planning', 'behavior', 'perception'):
            if qa.get(category):
                question = qa[category][0]['Q']
                break
        if not question:
            continue

        # Need 3 future samples (for t1, t2, t3)
        sample = nusc.get('sample', sample_token)
        future_tokens = []
        tok = sample['next']
        for _ in range(3):
            if tok == '':
                break
            future_tokens.append(tok)
            tok = nusc.get('sample', tok)['next']
        if len(future_tokens) < 3:
            continue

        scene_name = scene_token_to_name[scene_token]
        selected.append({
            'sample_token':  sample_token,
            'scene_token':   scene_token,
            'scene_name':    scene_name,
            'question':      question,
            'future_tokens': future_tokens[:3],
        })
        per_scene_counts[scene_name] = per_scene_counts.get(scene_name, 0) + 1

# ── Save ──────────────────────────────────────────────────────────────────────
import os
os.makedirs('results', exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(selected, f, indent=2)

print(f"\nSelected {len(selected)} keyframes across {len(per_scene_counts)} scenes.")
for name in TARGET_SCENES:
    n = per_scene_counts.get(name, 0)
    marker = "" if n > 0 else "  (no DriveLM annotations or no valid keyframes)"
    print(f"  {name}: {n} keyframes{marker}")
print(f"\nSaved to {OUTPUT_PATH}")
