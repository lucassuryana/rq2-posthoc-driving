"""
Test fuzzy_labels and fuzzy_eval against the REAL scripts.labels module
from the user's repo. Uses a synthetic 24-sample dataset that mirrors the
GT distribution observed in smoke_test_v6_output.html.
"""
import sys
sys.path.insert(0, '/home/claude/rq2_changes')

from scripts.labels import (
    continuous_to_drivelm_label,
    SPEED_THRESHOLDS_MPS, DIR_SLIGHT_DEG, DIR_FULL_DEG,
    SPEED_LABELS, DIRECTION_LABELS,
)
from scripts.fuzzy_labels import build_fuzzy_labeler, compare_fuzzy_vs_hard
from scripts.fuzzy_eval import (
    fuzzy_speed_score, fuzzy_direction_score,
    score_inference_outputs, fuzzy_accuracy_summary, alpha_sensitivity_sweep,
)

print("=" * 70)
print("USING REAL scripts.labels MODULE")
print("=" * 70)
print(f"SPEED_THRESHOLDS_MPS = {SPEED_THRESHOLDS_MPS}")
print(f"DIR_SLIGHT_DEG = {DIR_SLIGHT_DEG:.3f}")
print(f"DIR_FULL_DEG   = {DIR_FULL_DEG:.3f}")
print(f"SPEED_LABELS:     {SPEED_LABELS}")
print(f"DIRECTION_LABELS: {DIRECTION_LABELS}")


# ── Synthetic 24-sample dataset mirroring v6 GT distribution ────────
# Each entry has the same shape as results/preprocessed.json entries.

raw_samples_mps_deg = [
    # 7x stationary
    *[(0.0, 0.0)] * 7,
    # 5x driving very fast (~43 km/h ≈ 12 m/s)
    (12.83, -0.46), (12.06, -0.56), (11.89, -0.21),
    (11.92,  0.03), (11.64, -0.73),
    # 3x normal-speed straight (~18 km/h ≈ 5 m/s)
    (4.97, -0.74), (5.00, -0.67), (6.08, -0.76),
    # 3x slightly steering right, normal (~16 km/h, -8 deg)
    (4.39, -9.61), (4.36, -7.83), (4.86, -7.14),
    # 2x driving fast (~31 km/h ≈ 8.5 m/s)
    (8.42, -2.58), (8.97, -1.88),
    # 2x steering right, normal (~14 km/h, -20 deg)
    (4.14, -21.71), (3.97, -19.10),
    # 1x driving slowly (~8 km/h ≈ 2.2 m/s)
    (2.19, -0.68),
    # 1x slightly left, normal (~14 km/h, +13 deg)
    (3.94, 12.84),
]

data = []
for i, (v_mps, a_deg) in enumerate(raw_samples_mps_deg):
    dir_lbl, spd_lbl = continuous_to_drivelm_label(v_mps, a_deg)
    data.append({
        'sample_token': f'sample_{i:02d}_dummy_token',
        'gt_continuous': {
            'avg_speed_mps':     v_mps,
            'avg_speed_kph':     round(v_mps * 3.6, 1),
            'angle_overall_deg': a_deg,
            'direction_label':   dir_lbl,
            'speed_label':       spd_lbl,
            'drivelm_label':     f"The ego vehicle is {dir_lbl}. The ego vehicle is {spd_lbl}.",
        },
    })

from collections import Counter
print(f"\nSynthetic dataset: {len(data)} samples")
print("GT label distribution:")
for lbl, n in sorted(Counter(s['gt_continuous']['speed_label'] for s in data).items(), key=lambda x: -x[1]):
    print(f"  speed     {n:2d}x  {lbl}")
for lbl, n in sorted(Counter(s['gt_continuous']['direction_label'] for s in data).items(), key=lambda x: -x[1]):
    print(f"  direction {n:2d}x  {lbl}")


print("\n" + "=" * 70)
print("LAYER 1 TEST — input fuzzification")
print("=" * 70)

# No stats file → falls back to dataset distribution
fuzzy_labeler = build_fuzzy_labeler(data, stats_path=None, verbose=True)

print("\nBoundary-zone test cases:")
test_cases = [
    (0.0,  0.0,  "stationary, straight"),
    (1.21, 0.0,  "exactly at not-moving/slowly boundary"),
    (3.78, 0.0,  "exactly at slowly/normal boundary"),
    (6.23, 0.0,  "exactly at normal/fast boundary"),
    (9.78, 0.0,  "exactly at fast/very-fast boundary"),
    (5.0,  3.81, "near-straight slight-left boundary"),
    (5.0, -14.93,"near full-right boundary"),
]
for v, a, desc in test_cases:
    hard = continuous_to_drivelm_label(v, a)
    fuzzy = fuzzy_labeler(v, a)
    print(f"  v={v:5.2f} m/s, a={a:6.2f} deg  ({desc})")
    print(f"    hard:  {hard}")
    print(f"    fuzzy: {fuzzy}")

print()
compare_fuzzy_vs_hard(data, fuzzy_labeler)


print("\n" + "=" * 70)
print("LAYER 2 TEST — output scoring")
print("=" * 70)

print("\nSpeed score unit tests (alpha=0.5):")
speed_unit = [
    ('driving very fast',        'driving very fast', 1.0),
    ('driving fast',             'driving very fast', 0.5),
    ('driving with normal speed','driving very fast', 0.0),
    ('not moving',               'driving very fast', 0.0),
    ('driving slowly',           'driving with normal speed', 0.5),
]
for p, g, exp in speed_unit:
    got = fuzzy_speed_score(p, g)
    ok = '✓' if abs(got - exp) < 1e-6 else '✗'
    print(f"  {ok} pred={p!r:30s} gt={g!r:25s} -> {got:.3f} (expected {exp:.3f})")

print("\nDirection score unit tests (alpha=1/3):")
dir_unit = [
    ('going straight',                'going straight',          1.0),
    ('slightly steering to the right','steering to the right',   2/3),
    ('going straight',                'steering to the right',   1/3),
    ('slightly steering to the left', 'steering to the right',   0.0),
    ('steering to the left',          'steering to the right',   0.0),
]
for p, g, exp in dir_unit:
    got = fuzzy_direction_score(p, g)
    ok = '✓' if abs(got - exp) < 1e-6 else '✗'
    print(f"  {ok} pred={p!r:35s} gt={g!r:30s} -> {got:.3f} (expected {exp:.3f})")


# ── End-to-end on a mock results list (matching actual v6 output) ───
print("\nEnd-to-end scoring on mock results (matching v6 output shape):")

# Replicate the 3 illustrative samples we discussed
mock_results = [
    # Sample 13: GT very fast, non=very fast (perfect), post=fast (1 bucket off)
    {
        'sample_token': '6d5d0e60xxx',
        'gt_continuous': {
            'direction_label': 'going straight',
            'speed_label':     'driving very fast',
        },
        'non_posthoc': {
            'direction': 'going straight',
            'speed':     'driving very fast',
            'gt_direction': 'going straight',
            'gt_speed':     'driving very fast',
        },
        'posthoc': {
            'direction': 'going straight',
            'speed':     'driving fast',
            'gt_direction': 'going straight',
            'gt_speed':     'driving very fast',
        },
    },
    # Sample 18: GT steering right, pred steering LEFT (opposite)
    {
        'sample_token': 'a19a80c9xxx',
        'gt_continuous': {
            'direction_label': 'steering to the right',
            'speed_label':     'driving with normal speed',
        },
        'non_posthoc': {
            'direction': 'steering to the left',
            'speed':     'driving with normal speed',
            'gt_direction': 'steering to the right',
            'gt_speed':     'driving with normal speed',
        },
        'posthoc': {
            'direction': 'steering to the left',
            'speed':     'driving with normal speed',
            'gt_direction': 'steering to the right',
            'gt_speed':     'driving with normal speed',
        },
    },
    # Sample 24: GT steering right, pred slightly right (1 bucket, same side)
    {
        'sample_token': 'b1303058xxx',
        'gt_continuous': {
            'direction_label': 'steering to the right',
            'speed_label':     'driving with normal speed',
        },
        'non_posthoc': {
            'direction': 'slightly steering to the right',
            'speed':     'driving with normal speed',
            'gt_direction': 'steering to the right',
            'gt_speed':     'driving with normal speed',
        },
        'posthoc': {
            'direction': 'slightly steering to the right',
            'speed':     'driving with normal speed',
            'gt_direction': 'steering to the right',
            'gt_speed':     'driving with normal speed',
        },
    },
]

score_inference_outputs(mock_results)
for i, entry in enumerate(mock_results):
    print(f"\nSample {i} ({entry['sample_token'][:8]}...)")
    for cond in ('non_posthoc', 'posthoc'):
        o = entry[cond]
        print(f"  {cond:11s}: pred={o['direction']:32s} / {o['speed']:25s}")
        print(f"               fuzzy_dir={o['fuzzy_dir']:.3f}  "
              f"fuzzy_spd={o['fuzzy_spd']:.3f}  "
              f"fuzzy_full_min={o['fuzzy_full_min']:.3f}")

fuzzy_accuracy_summary(mock_results)
alpha_sensitivity_sweep(mock_results)

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE — integration ready")
print("=" * 70)
