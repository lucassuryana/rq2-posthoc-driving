"""
scripts/02b_add_gt_continuous.py
Adds gt_continuous field to preprocessed.json.

Computes:
  - Per-step instantaneous speeds (speed_1, speed_2, speed_3)
  - Average speed and acceleration
  - Per-step heading angles (angle_1, angle_2, angle_3)
  - Overall heading angle (t0 to t3)
  - Yaw rate (rad/s and deg/s)
  - DriveLM-style categorical labels derived from the continuous fields

Trajectory stored relative to t0:
  traj[0] = {x,y} at t+0.5s
  traj[1] = {x,y} at t+1.0s
  traj[2] = {x,y} at t+1.5s
"""
import json, math

with open('results/preprocessed.json') as f:
    preprocessed = json.load(f)


import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import continuous_to_drivelm_label, format_drivelm_label


def derive_gt_continuous(traj):
    x0, y0 = 0.0, 0.0
    x1, y1 = traj[0]['x'], traj[0]['y']
    x2, y2 = traj[1]['x'], traj[1]['y']
    x3, y3 = traj[2]['x'], traj[2]['y']

    # ── Step distances ────────────────────────────────────────────
    d1 = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    d2 = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    d3 = math.sqrt((x3-x2)**2 + (y3-y2)**2)

    # ── Speed ────────────────────────────────────────────────────
    speed_1   = d1 / 0.5
    speed_2   = d2 / 0.5
    speed_3   = d3 / 0.5
    avg_speed = (speed_1 + speed_2 + speed_3) / 3
    acceleration = (speed_3 - speed_1) / 1.0

    # ── Per-step heading angles ───────────────────────────────────
    angle_1_rad = math.atan2(y1-y0, x1-x0) if d1 > 0.1 else 0.0
    angle_2_rad = math.atan2(y2-y1, x2-x1) if d2 > 0.1 else 0.0
    angle_3_rad = math.atan2(y3-y2, x3-x2) if d3 > 0.1 else 0.0

    # ── Overall heading angle (t0 to t3) ─────────────────────────
    total_dist = math.sqrt(x3**2 + y3**2)
    angle_overall_rad = 0.0 if total_dist < 0.1 else math.atan2(y3, x3)
    angle_overall_deg = math.degrees(angle_overall_rad)

    # ── Yaw rate ─────────────────────────────────────────────────
    yaw_rate_rad = (angle_3_rad - angle_1_rad) / 1.0
    yaw_rate_deg = math.degrees(yaw_rate_rad)

    # ── DriveLM categorical label (derived from continuous fields) ─
    direction_label, speed_label = continuous_to_drivelm_label(
        avg_speed, angle_overall_deg
    )
    drivelm_label = (
        f"The ego vehicle is {direction_label}. "
        f"The ego vehicle is {speed_label}."
    )

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
    }


# ── Run + verify against existing gt_label ────────────────────────
print(f"{'Sample':<10} {'kph':>6} {'angle°':>7}  "
      f"{'derived label':<60} {'match':>5}")
print("-" * 95)

n_match_full = n_match_dir = n_match_spd = 0
for entry in preprocessed:
    gt = derive_gt_continuous(entry['trajectory'])
    entry['gt_continuous'] = gt

    actual = entry.get('gt_label', '')
    dir_ok = gt['direction_label'] in actual
    spd_ok = gt['speed_label']     in actual
    n_match_dir  += dir_ok
    n_match_spd  += spd_ok
    n_match_full += (dir_ok and spd_ok)

    mark = "✓" if (dir_ok and spd_ok) else ("~" if (dir_ok or spd_ok) else "✗")
    print(f"{entry['sample_token'][:8]:<10} "
          f"{gt['avg_speed_kph']:>6.1f} "
          f"{gt['angle_overall_deg']:>7.2f}  "
          f"{gt['drivelm_label'][:58]:<60} {mark:>5}")

n = len(preprocessed)
print(f"\nDirection match: {n_match_dir}/{n} = {n_match_dir/n:.1%}")
print(f"Speed match:     {n_match_spd}/{n} = {n_match_spd/n:.1%}")
print(f"Full match:      {n_match_full}/{n} = {n_match_full/n:.1%}")

with open('results/preprocessed.json', 'w') as f:
    json.dump(preprocessed, f, indent=2)
print("\nSaved to results/preprocessed.json")
