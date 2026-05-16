"""
scripts/02b_add_gt_continuous_target.py

Same as 02b_add_gt_continuous.py, but reads/writes the target preprocessed
file and loads nuScenes v1.0-trainval.
"""
import json, math, os, sys
from nuscenes.nuscenes import NuScenes

INPUT_FILE  = 'results/preprocessed_target.json'
OUTPUT_FILE = 'results/preprocessed_target.json'   # overwrite in place
NUSC_ROOT   = '/tudelft.net/staff-umbrella/lsuryana/rq2-posthoc-driving/data/nuscenes'
NUSC_VER    = 'v1.0-trainval'

with open(INPUT_FILE) as f:
    preprocessed = json.load(f)

nusc = NuScenes(version=NUSC_VER, dataroot=NUSC_ROOT, verbose=False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import continuous_to_drivelm_label, format_drivelm_label, DIR_SLIGHT_DEG, DIR_FULL_DEG


def get_location(scene_token):
    scene = nusc.get('scene', scene_token)
    log   = nusc.get('log', scene['log_token'])
    loc   = log['location']
    if loc.startswith('boston'):
        return {'location': loc, 'city': 'Boston',
                'country': 'USA', 'drive_side': 'right'}
    elif loc.startswith('singapore'):
        return {'location': loc, 'city': 'Singapore',
                'country': 'Singapore', 'drive_side': 'left'}
    else:
        return {'location': loc, 'city': loc,
                'country': 'unknown', 'drive_side': 'unknown'}


def derive_ego_goal(trajectory, angle_overall_deg):
    total_dist = math.sqrt(trajectory[2]['x']**2 + trajectory[2]['y']**2)
    if total_dist < 1.0:
        return "remain stopped and wait"
    elif angle_overall_deg >  DIR_FULL_DEG:
        return "turn left at the upcoming junction"
    elif angle_overall_deg >  DIR_SLIGHT_DEG:
        return "bear left along the road"
    elif angle_overall_deg < -DIR_FULL_DEG:
        return "turn right at the upcoming junction"
    elif angle_overall_deg < -DIR_SLIGHT_DEG:
        return "bear right along the road"
    else:
        return "continue straight along the current road"


def derive_gt_continuous(traj):
    x0, y0 = 0.0, 0.0
    x1, y1 = traj[0]['x'], traj[0]['y']
    x2, y2 = traj[1]['x'], traj[1]['y']
    x3, y3 = traj[2]['x'], traj[2]['y']
    d1 = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    d2 = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    d3 = math.sqrt((x3-x2)**2 + (y3-y2)**2)
    speed_1 = d1 / 0.5
    speed_2 = d2 / 0.5
    speed_3 = d3 / 0.5
    avg_speed = (speed_1 + speed_2 + speed_3) / 3
    acceleration = (speed_3 - speed_1) / 1.0
    angle_1_rad = math.atan2(y1-y0, x1-x0) if d1 > 0.1 else 0.0
    angle_2_rad = math.atan2(y2-y1, x2-x1) if d2 > 0.1 else 0.0
    angle_3_rad = math.atan2(y3-y2, x3-x2) if d3 > 0.1 else 0.0
    total_dist = math.sqrt(x3**2 + y3**2)
    angle_overall_rad = 0.0 if total_dist < 0.1 else math.atan2(y3, x3)
    angle_overall_deg = math.degrees(angle_overall_rad)
    yaw_rate_rad = (angle_3_rad - angle_1_rad) / 1.0
    yaw_rate_deg = math.degrees(yaw_rate_rad)
    direction_label, speed_label = continuous_to_drivelm_label(
        avg_speed, angle_overall_deg)
    drivelm_label = (f"The ego vehicle is {direction_label}. "
                     f"The ego vehicle is {speed_label}.")
    return {
        'speed_1_mps':       round(speed_1, 3),
        'speed_2_mps':       round(speed_2, 3),
        'speed_3_mps':       round(speed_3, 3),
        'avg_speed_mps':     round(avg_speed, 3),
        'avg_speed_kph':     round(avg_speed * 3.6, 1),
        'acceleration_mps2': round(acceleration, 3),
        'angle_1_rad':       round(angle_1_rad, 4),
        'angle_1_deg':       round(math.degrees(angle_1_rad), 2),
        'angle_2_rad':       round(angle_2_rad, 4),
        'angle_2_deg':       round(math.degrees(angle_2_rad), 2),
        'angle_3_rad':       round(angle_3_rad, 4),
        'angle_3_deg':       round(math.degrees(angle_3_rad), 2),
        'angle_overall_rad': round(angle_overall_rad, 4),
        'angle_overall_deg': round(angle_overall_deg, 2),
        'yaw_rate_rad_s':    round(yaw_rate_rad, 4),
        'yaw_rate_deg_s':    round(yaw_rate_deg, 2),
        'direction_label':   direction_label,
        'speed_label':       speed_label,
        'drivelm_label':     drivelm_label,
        'ego_goal':          derive_ego_goal(traj, angle_overall_deg),
    }


print(f"{'Scene':<13} {'Sample':<10} {'kph':>6} {'angle°':>7}  {'derived label':<60}")
print("-" * 105)

for entry in preprocessed:
    gt = derive_gt_continuous(entry['trajectory'])
    gt.update(get_location(entry['scene_token']))
    entry['gt_continuous'] = gt
    entry['ego_goal'] = gt['ego_goal']
    print(f"{entry.get('scene_name','?'):<13} "
          f"{entry['sample_token'][:8]:<10} "
          f"{gt['avg_speed_kph']:>6.1f} "
          f"{gt['angle_overall_deg']:>7.2f}  "
          f"{gt['drivelm_label'][:58]:<60}")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(preprocessed, f, indent=2)
print(f"\nSaved {len(preprocessed)} samples to {OUTPUT_FILE}")
