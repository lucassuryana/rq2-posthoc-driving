"""
scripts/fuzzy_labels.py
Fuzzy Layer 1 — input-side fuzzification of past behavior labels.

Replaces scripts.labels.continuous_to_drivelm_label with a Mamdani-style
fuzzy classifier using trapezoidal membership functions and argmax
defuzzification. Returns the same (direction, speed) tuple of DriveLM
vocabulary strings — drop-in compatible.

Boundary derivation is consistent with scripts/04_drivelm_label_stats.py:
each fuzzy set has a plateau [p25, p75] of its own class samples, and feet
that extend to the 75th percentile of the previous class and the 25th
percentile of the next class. Sparse classes (n < MIN_SAMPLES_PER_LABEL)
fall back to the original hard thresholds from scripts.labels.

Usage:
    from scripts.fuzzy_labels import build_fuzzy_labeler

    fuzzy_labeler = build_fuzzy_labeler(data)
    direction, speed = fuzzy_labeler(avg_speed_mps, angle_overall_deg)
"""
import os
import json
import numpy as np

from scripts.labels import (
    continuous_to_drivelm_label,
    SPEED_LABELS,
    DIRECTION_LABELS,
    SPEED_THRESHOLDS_MPS,
    DIR_SLIGHT_DEG,
    DIR_FULL_DEG,
)


MIN_SAMPLES_PER_LABEL = 3
DEFAULT_STATS_PATH    = 'config/drivelm_label_stats.json'


# ── Trapezoidal membership function ──────────────────────────────────

def trapezoid(x, a, b, c, d):
    """
    Trapezoidal membership with parameters a <= b <= c <= d.
    Rises a -> b, plateau b -> c, falls c -> d. Zero outside [a, d].
    """
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b > a else 1.0
    if c < x < d:
        return (d - x) / (d - c) if d > c else 1.0
    return 0.0


# ── Boundary derivation ──────────────────────────────────────────────

def _derive_boundaries_from_distribution(values_by_label, label_order):
    """
    For each label, build trapezoid parameters (a, b, c, d) from its
    sample distribution and adjacent classes:
      [b, c] is the plateau (p25 to p75 of own samples)
      a     is p75 of the nearest non-sparse previous label
      d     is p25 of the nearest non-sparse next label
    """
    boundaries = {}
    for i, lbl in enumerate(label_order):
        own = values_by_label.get(lbl, [])
        if len(own) < MIN_SAMPLES_PER_LABEL:
            continue
        own_arr = np.array(own, dtype=float)
        b = float(np.percentile(own_arr, 25))
        c = float(np.percentile(own_arr, 75))

        a = None
        for j in range(i - 1, -1, -1):
            prev = values_by_label.get(label_order[j], [])
            if len(prev) >= MIN_SAMPLES_PER_LABEL:
                a = float(np.percentile(prev, 75))
                break
        if a is None:
            a = float(own_arr.min())

        d = None
        for j in range(i + 1, len(label_order)):
            nxt = values_by_label.get(label_order[j], [])
            if len(nxt) >= MIN_SAMPLES_PER_LABEL:
                d = float(np.percentile(nxt, 25))
                break
        if d is None:
            d = float(own_arr.max())

        a = min(a, b)
        d = max(d, c)
        boundaries[lbl] = (a, b, c, d)
    return boundaries


def _derive_boundaries_from_stats_file(stats_path):
    """
    Reuse pre-computed stats from 04_drivelm_label_stats.py.
    Returns (speed_boundaries, dir_boundaries).
    """
    if not os.path.exists(stats_path):
        return None, None
    with open(stats_path) as f:
        stats = json.load(f)

    def build(stats_dict, label_order, phys_key):
        boundaries = {}
        plateaus = {}
        for lbl in label_order:
            s = stats_dict.get(lbl, {})
            if s.get('n', 0) < MIN_SAMPLES_PER_LABEL:
                continue
            d = s[phys_key]
            plateaus[lbl] = (d['p25'], d['p75'])

        for i, lbl in enumerate(label_order):
            if lbl not in plateaus:
                continue
            b, c = plateaus[lbl]

            a = None
            for j in range(i - 1, -1, -1):
                prev = stats_dict.get(label_order[j], {})
                if prev.get('n', 0) >= MIN_SAMPLES_PER_LABEL:
                    a = prev[phys_key]['p75']
                    break
            if a is None:
                a = stats_dict[lbl][phys_key]['min']

            d_param = None
            for j in range(i + 1, len(label_order)):
                nxt = stats_dict.get(label_order[j], {})
                if nxt.get('n', 0) >= MIN_SAMPLES_PER_LABEL:
                    d_param = nxt[phys_key]['p25']
                    break
            if d_param is None:
                d_param = stats_dict[lbl][phys_key]['max']

            a = min(a, b)
            d_param = max(d_param, c)
            boundaries[lbl] = (a, b, c, d_param)
        return boundaries

    speed_bounds = build(stats['speed_stats'],     SPEED_LABELS,     'avg_speed_mps')
    dir_bounds   = build(stats['direction_stats'], DIRECTION_LABELS, 'angle_overall_deg')
    return speed_bounds, dir_bounds


# ── Public builder ───────────────────────────────────────────────────

def build_fuzzy_labeler(data=None, stats_path=DEFAULT_STATS_PATH, verbose=True):
    """
    Build a fuzzy labeler closure.

    Boundary source priority:
      1. If `stats_path` exists, use its richer DriveLM∩nuScenes stats
         (recommended — derived from n=38 in 04_drivelm_label_stats.py).
      2. Otherwise, derive boundaries from `data` distribution.
      3. For any class still missing, fall back to hard thresholds in
         scripts.labels at runtime.

    Returns:
        callable(avg_speed_mps, angle_deg, horizon_s=1.5) -> (direction, speed)
    """
    speed_bounds, dir_bounds = None, None
    source = None

    if stats_path and os.path.exists(stats_path):
        try:
            speed_bounds, dir_bounds = _derive_boundaries_from_stats_file(stats_path)
            source = f'stats file: {stats_path}'
        except (KeyError, json.JSONDecodeError) as e:
            if verbose:
                print(f"[fuzzy_labels] Could not read {stats_path}: {e}")
            speed_bounds, dir_bounds = None, None

    if speed_bounds is None or dir_bounds is None:
        if data is None:
            raise ValueError(
                "No usable stats_path and no data provided — "
                "cannot derive fuzzy boundaries."
            )
        speed_vals, angle_vals = {}, {}
        for s in data:
            gt = s.get('gt_continuous') or {}
            v = gt.get('avg_speed_mps')
            a = gt.get('angle_overall_deg')
            sl = gt.get('speed_label')
            dl = gt.get('direction_label')
            if v is not None and sl:
                speed_vals.setdefault(sl, []).append(v)
            if a is not None and dl:
                angle_vals.setdefault(dl, []).append(a)
        speed_bounds = _derive_boundaries_from_distribution(speed_vals, SPEED_LABELS)
        dir_bounds   = _derive_boundaries_from_distribution(angle_vals, DIRECTION_LABELS)
        source = f'dataset distribution (n={len(data)})'

    if verbose:
        print(f"Fuzzy Layer 1 — boundary source: {source}\n")
        print("Speed (avg_speed_mps):")
        for lbl in SPEED_LABELS:
            if lbl in speed_bounds:
                a, b, c, d = speed_bounds[lbl]
                print(f"  {lbl:30s}  trap=({a:5.2f}, {b:5.2f}, {c:5.2f}, {d:5.2f})")
            else:
                print(f"  {lbl:30s}  [SPARSE — hard-threshold fallback]")
        print("\nDirection (angle_overall_deg):")
        for lbl in DIRECTION_LABELS:
            if lbl in dir_bounds:
                a, b, c, d = dir_bounds[lbl]
                print(f"  {lbl:35s}  trap=({a:6.2f}, {b:6.2f}, {c:6.2f}, {d:6.2f})")
            else:
                print(f"  {lbl:35s}  [SPARSE — hard-threshold fallback]")
        print(f"\nHard-threshold fallback values: "
              f"speed_thresholds_mps={SPEED_THRESHOLDS_MPS}, "
              f"slight={DIR_SLIGHT_DEG:.2f}deg, full={DIR_FULL_DEG:.2f}deg")

    def fuzzy_labeler(avg_speed_mps, angle_deg, horizon_s=1.5):
        """
        Drop-in replacement for continuous_to_drivelm_label.
        Returns (direction, speed) — same order, same vocabulary.
        """
        spd = _classify(avg_speed_mps, speed_bounds, SPEED_LABELS)
        drc = _classify(angle_deg,     dir_bounds,   DIRECTION_LABELS)

        if spd is None or drc is None:
            hard_drc, hard_spd = continuous_to_drivelm_label(
                avg_speed_mps, angle_deg, horizon_s=horizon_s)
            if spd is None: spd = hard_spd
            if drc is None: drc = hard_drc

        return drc, spd

    return fuzzy_labeler


def _classify(x, boundaries, label_order):
    """Compute memberships and return argmax. None if all zero."""
    memberships = {lbl: trapezoid(x, *params)
                   for lbl, params in boundaries.items()}
    if not memberships or max(memberships.values()) == 0.0:
        return None

    best_lbl, best_mem, best_idx = None, -1.0, len(label_order)
    for lbl, mem in memberships.items():
        idx = label_order.index(lbl)
        if mem > best_mem or (mem == best_mem and idx < best_idx):
            best_lbl, best_mem, best_idx = lbl, mem, idx
    return best_lbl


# ── Diagnostic ───────────────────────────────────────────────────────

def compare_fuzzy_vs_hard(data, fuzzy_labeler):
    """
    Run both labelers on every GT continuous motion and report agreement.
    """
    agreements = 0
    disagreements = []
    for s in data:
        gt = s.get('gt_continuous') or {}
        v_mps = gt.get('avg_speed_mps', 0.0)
        a_deg = gt.get('angle_overall_deg', 0.0)
        tok   = s.get('sample_token', '????????')[:8]

        hard  = continuous_to_drivelm_label(v_mps, a_deg)
        fuzzy = fuzzy_labeler(v_mps, a_deg)

        if hard == fuzzy:
            agreements += 1
        else:
            disagreements.append((tok, v_mps, a_deg, hard, fuzzy))

    print(f"\nFuzzy vs hard labeling agreement: {agreements}/{len(data)}")
    if disagreements:
        print("Disagreements:")
        for tok, v, a, h, f in disagreements:
            print(f"  {tok}: v={v:.2f}m/s, a={a:.2f}deg")
            print(f"     hard:  {h}")
            print(f"     fuzzy: {f}")
    return agreements, disagreements
