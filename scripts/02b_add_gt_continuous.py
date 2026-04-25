"""
scripts/02b_add_gt_continuous.py
Adds gt_continuous field to preprocessed.json.
Computes speed (m/s, km/h) and heading angle (rad, deg) from ego trajectory.
"""
import json, math

with open('results/preprocessed.json') as f:
    preprocessed = json.load(f)

def derive_gt_continuous(traj):
    x3 = traj[2]['x']
    y3 = traj[2]['y']
    dist  = math.sqrt(x3**2 + y3**2)
    speed = dist / 1.5
    angle_rad = 0.0 if dist < 0.1 else math.atan2(y3, x3)
    return {
        'speed_mps': round(speed, 3),
        'speed_kph': round(speed * 3.6, 1),
        'angle_rad': round(angle_rad, 4),
        'angle_deg': round(math.degrees(angle_rad), 2),
    }

print(f"{'Sample':<12} {'x3':>7} {'y3':>7} {'speed_mps':>10} {'speed_kph':>10} {'angle_deg':>10}")
print("-"*60)

for entry in preprocessed:
    traj = entry['trajectory']
    gt_cont = derive_gt_continuous(traj)
    entry['gt_continuous'] = gt_cont
    print(f"{entry['sample_token'][:8]:<12} "
          f"{traj[2]['x']:>7.3f} {traj[2]['y']:>7.3f} "
          f"{gt_cont['speed_mps']:>10.3f} "
          f"{gt_cont['speed_kph']:>10.1f} "
          f"{gt_cont['angle_deg']:>10.2f}")

with open('results/preprocessed.json', 'w') as f:
    json.dump(preprocessed, f, indent=2)
print("\nSaved to results/preprocessed.json")
