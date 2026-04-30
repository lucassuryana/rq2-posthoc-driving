"""
scripts/labels.py
Single source of truth for DriveLM categorical labelling.

Speed thresholds: empirical, derived from DriveLM ∩ nuScenes mini (n=38)
                  via scripts/04_drivelm_label_stats.py.
Direction thresholds: DriveLM-published y3 thresholds converted to angle
                  equivalents at typical urban speed (5 m/s × 1.5 s = 7.5 m).
                  Direction not empirically derivable from mini due to
                  sparse coverage (only 2 of 5 classes populated).
"""
import math

# ── Speed thresholds on avg_speed_mps (m/s) ──────────────────────────
# Empirical class boundaries from DriveLM ∩ nuScenes mini, n=38.
# Each value is the boundary BELOW which the class applies.
SPEED_THRESHOLDS_MPS = [1.21, 3.78, 6.23, 9.78]
SPEED_LABELS = [
    'not moving',
    'driving slowly',
    'driving with normal speed',
    'driving fast',
    'driving very fast',
]

# ── Direction thresholds on angle_overall_deg ────────────────────────
# Derived from DriveLM-published y3 thresholds (±0.5 m, ±2.0 m) at the
# typical 1.5-second horizon distance of 7.5 m:
#   slight: |angle| > atan2(0.5, 7.5) = 3.81°
#   full:   |angle| > atan2(2.0, 7.5) = 14.93°
DIR_SLIGHT_DEG = math.degrees(math.atan2(0.5, 7.5))   # ≈ 3.81
DIR_FULL_DEG   = math.degrees(math.atan2(2.0, 7.5))   # ≈ 14.93
DIRECTION_LABELS = [
    'steering to the left',
    'slightly steering to the left',
    'going straight',
    'slightly steering to the right',
    'steering to the right',
]


def continuous_to_drivelm_label(speed_mps, angle_deg, horizon_s=1.5):
    """
    Map continuous physics fields to DriveLM categorical labels.
    Applies thresholds directly to speed and angle — no x3/y3 reconstruction.

    Returns (direction, speed) as DriveLM-vocabulary strings.
    """
    # Direction: thresholded directly on angle_overall_deg
    if   angle_deg >  DIR_FULL_DEG:   direction = 'steering to the left'
    elif angle_deg >  DIR_SLIGHT_DEG: direction = 'slightly steering to the left'
    elif angle_deg < -DIR_FULL_DEG:   direction = 'steering to the right'
    elif angle_deg < -DIR_SLIGHT_DEG: direction = 'slightly steering to the right'
    else:                             direction = 'going straight'

    # Speed: thresholded directly on avg_speed_mps
    if   speed_mps < SPEED_THRESHOLDS_MPS[0]: speed = 'not moving'
    elif speed_mps < SPEED_THRESHOLDS_MPS[1]: speed = 'driving slowly'
    elif speed_mps < SPEED_THRESHOLDS_MPS[2]: speed = 'driving with normal speed'
    elif speed_mps < SPEED_THRESHOLDS_MPS[3]: speed = 'driving fast'
    else:                                    speed = 'driving very fast'

    return direction, speed


def format_drivelm_label(direction, speed):
    """Combine direction + speed into the canonical DriveLM sentence."""
    return f"The ego vehicle is {direction}. The ego vehicle is {speed}."