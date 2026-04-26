"""
scripts/02b_add_gt_continuous.py
Adds gt_continuous and prev_continuous fields to preprocessed.json.

gt_continuous:   computed from future trajectory (t0 to t+1.5s)
prev_continuous: computed from past trajectory (t-1.5s to t0)

Both compute:
  - Per-step instantaneous speeds
  - Average speed and acceleration
  - Per-step heading angles
  - Overall heading angle
  - Yaw rate
"""
import json, math

with open('results/preprocessed.json') as f:
    preprocessed = json.load(f)

def derive_continuous(p0, p1, p2, p3):
    """
    Compute continuous motion metrics from 4 positions.
    p0 = reference point (origin)
    p1, p2, p3 = subsequent positions relative to p0
    Each step = 0.5s apart.

    For future: p0=(0,0), p1=t+0.5s, p2=t+1.0s, p3=t+1.5s
    For past:   p0=(0,0), p1=t-1.0s, p2=t-0.5s, p3=t0 (all relative to t0)
                Note: past positions have negative x (behind t0)
    """
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Step distances
    d1 = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    d2 = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    d3 = math.sqrt((x3-x2)**2 + (y3-y2)**2)

    # Speeds
    speed_1   = d1 / 0.5
    speed_2   = d2 / 0.5
    speed_3   = d3 / 0.5
    avg_speed = (speed_1 + speed_2 + speed_3) / 3
    acceleration = (speed_3 - speed_1) / 1.0

    # Per-step angles
    angle_1_rad = math.atan2(y1-y0, x1-x0) if d1 > 0.1 else 0.0
    angle_2_rad = math.atan2(y2-y1, x2-x1) if d2 > 0.1 else 0.0
    angle_3_rad = math.atan2(y3-y2, x3-x2) if d3 > 0.1 else 0.0

    # Overall angle
    total_dist = math.sqrt((x3-x0)**2 + (y3-y0)**2)
    angle_overall_rad = 0.0 if total_dist < 0.1 else math.atan2(y3-y0, x3-x0)

    # Yaw rate
    yaw_rate_rad = (angle_3_rad - angle_1_rad) / 1.0
    yaw_rate_deg = math.degrees(yaw_rate_rad)

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
        'angle_overall_deg': round(math.degrees(angle_overall_rad), 2),
        'yaw_rate_rad_s':    round(yaw_rate_rad, 4),
        'yaw_rate_deg_s':    round(yaw_rate_deg, 2),
    }

print("Computing gt_continuous (future) and prev_continuous (past)...\n")
print(f"{'Sample':<12} {'FUTURE avg_kph':>14} {'FUTURE angle°':>13} "
      f"{'PAST avg_kph':>12} {'PAST angle°':>11} {'PAST yaw°/s':>11}")
print("-"*80)

for entry in preprocessed:
    traj      = entry['trajectory']
    past_traj = entry.get('past_trajectory', [])
    n_past    = entry.get('n_past', 0)

    # ── GT continuous (future: t0 → t+1.5s) ──────────────────────
    gt = derive_continuous(
        (0.0, 0.0),
        (traj[0]['x'], traj[0]['y']),
        (traj[1]['x'], traj[1]['y']),
        (traj[2]['x'], traj[2]['y']),
    )
    entry['gt_continuous'] = gt

    # ── Prev continuous (past: t-1.5s → t0) ──────────────────────
    # past_traj is ordered [t-3, t-2, t-1] relative to t0
    # positions have negative x (behind t0)
    # We reverse to get chronological order: earliest → t0
    if n_past >= 3:
        # Use all 3 past frames: t-3→t-2, t-2→t-1, t-1→t0
        prev = derive_continuous(
            (past_traj[0]['x'], past_traj[0]['y']),  # t-3 (earliest)
            (past_traj[1]['x'], past_traj[1]['y']),  # t-2
            (past_traj[2]['x'], past_traj[2]['y']),  # t-1
            (0.0, 0.0),                               # t0
        )
    elif n_past == 2:
        # Only 2 past frames: use t-2, t-1, t0
        # Pad with duplicate of first point
        prev = derive_continuous(
            (past_traj[0]['x'], past_traj[0]['y']),  # t-2
            (past_traj[0]['x'], past_traj[0]['y']),  # t-2 (duplicated)
            (past_traj[1]['x'], past_traj[1]['y']),  # t-1
            (0.0, 0.0),                               # t0
        )
    else:
        prev = None

    entry['prev_continuous'] = prev

    prev_kph = prev['avg_speed_kph'] if prev else 0.0
    prev_ang = prev['angle_overall_deg'] if prev else 0.0
    prev_yaw = prev['yaw_rate_deg_s'] if prev else 0.0

    print(f"{entry['sample_token'][:8]:<12} "
          f"{gt['avg_speed_kph']:>14.1f} "
          f"{gt['angle_overall_deg']:>13.2f} "
          f"{prev_kph:>12.1f} "
          f"{prev_ang:>11.2f} "
          f"{prev_yaw:>11.2f}")

with open('results/preprocessed.json', 'w') as f:
    json.dump(preprocessed, f, indent=2)
print("\nSaved to results/preprocessed.json")
