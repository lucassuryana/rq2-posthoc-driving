"""
scripts/02b_add_gt_continuous.py
Adds gt_continuous field to preprocessed.json.

Computes:
  - Per-step instantaneous speeds (speed_1, speed_2, speed_3)
  - Average speed and acceleration
  - Per-step heading angles (angle_1, angle_2, angle_3)
  - Overall heading angle (t0 to t3)
  - Yaw rate (how fast the vehicle is turning, rad/s and deg/s)

Trajectory stored relative to t0:
  traj[0] = {x,y} at t+0.5s
  traj[1] = {x,y} at t+1.0s
  traj[2] = {x,y} at t+1.5s
"""
import json, math

with open('results/preprocessed.json') as f:
    preprocessed = json.load(f)

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
    acceleration = (speed_3 - speed_1) / 1.0  # m/s²

    # ── Per-step heading angles ───────────────────────────────────
    # angle_1: direction of t0→t1
    # angle_2: direction of t1→t2
    # angle_3: direction of t2→t3
    angle_1_rad = math.atan2(y1-y0, x1-x0) if d1 > 0.1 else 0.0
    angle_2_rad = math.atan2(y2-y1, x2-x1) if d2 > 0.1 else 0.0
    angle_3_rad = math.atan2(y3-y2, x3-x2) if d3 > 0.1 else 0.0

    # ── Overall heading angle (t0 to t3) ─────────────────────────
    total_dist = math.sqrt(x3**2 + y3**2)
    angle_overall_rad = 0.0 if total_dist < 0.1 else math.atan2(y3, x3)

    # ── Yaw rate ─────────────────────────────────────────────────
    # Central difference: (angle_3 - angle_1) / 1.0s
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

print(f"{'Sample':<12} {'spd1':>5} {'spd2':>5} {'spd3':>5} "
      f"{'avg_mps':>7} {'avg_kph':>7} {'accel':>7} "
      f"{'a1°':>7} {'a2°':>7} {'a3°':>7} {'aOvr°':>7} {'yaw°/s':>7}")
print("-"*98)

for entry in preprocessed:
    traj = entry['trajectory']
    gt   = derive_gt_continuous(traj)
    entry['gt_continuous'] = gt
    print(f"{entry['sample_token'][:8]:<12} "
          f"{gt['speed_1_mps']:>5.2f} "
          f"{gt['speed_2_mps']:>5.2f} "
          f"{gt['speed_3_mps']:>5.2f} "
          f"{gt['avg_speed_mps']:>7.3f} "
          f"{gt['avg_speed_kph']:>7.1f} "
          f"{gt['acceleration_mps2']:>7.3f} "
          f"{gt['angle_1_deg']:>7.2f} "
          f"{gt['angle_2_deg']:>7.2f} "
          f"{gt['angle_3_deg']:>7.2f} "
          f"{gt['angle_overall_deg']:>7.2f} "
          f"{gt['yaw_rate_deg_s']:>7.2f}")

with open('results/preprocessed.json', 'w') as f:
    json.dump(preprocessed, f, indent=2)
print("\nSaved to results/preprocessed.json")
