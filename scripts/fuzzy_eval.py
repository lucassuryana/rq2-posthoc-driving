"""
scripts/fuzzy_eval.py
Fuzzy Layer 2 — output-side ordinal scoring for evaluation.

Replaces hard equality (direction == gt_direction) with graded scoring over
the ordered DriveLM label space. Adjacent-bucket errors get partial credit;
far errors get zero.

Score formula:
    d        = |position(pred) - position(gt)|
    score    = max(0, 1 - alpha * d)
    full_min = min(direction_score, speed_score)        # strict (weakest link)
    full_prod = direction_score * speed_score           # multiplicative

Default slopes:
    alpha_speed     = 0.5    (5 buckets, score reaches 0 at d=2)
    alpha_direction = 1/3    (5 buckets symmetric, score reaches 0 at d=3)

Usage:
    from scripts.fuzzy_eval import (
        score_inference_outputs, fuzzy_accuracy_summary,
        alpha_sensitivity_sweep,
    )

    score_inference_outputs(results)
    fuzzy_accuracy_summary(results)
    alpha_sensitivity_sweep(results)
"""
from scripts.labels import SPEED_LABELS, DIRECTION_LABELS


# ── Label position assignments ───────────────────────────────────────
# Built from scripts.labels orderings — translation-invariant: only
# differences between positions matter.

SPEED_POSITION     = {lbl: i for i, lbl in enumerate(SPEED_LABELS)}
DIRECTION_POSITION = {lbl: i - 2 for i, lbl in enumerate(DIRECTION_LABELS)}
# DIRECTION_POSITION: steering left = -2 ... steering right = +2

ALPHA_SPEED_DEFAULT     = 0.5       # d=1 -> 0.5, d>=2 -> 0
ALPHA_DIRECTION_DEFAULT = 1.0 / 3   # d=1 -> 0.67, d=2 -> 0.33, d>=3 -> 0


# ── Core scoring functions ───────────────────────────────────────────

def _ordinal_score(pred, gt, position_map, alpha):
    """
    Generic ordinal soft score. Returns 0.0 if either label is unknown
    (so parse failures are penalized as a miss, not silently scored 1.0).
    """
    if pred not in position_map or gt not in position_map:
        return 0.0
    d = abs(position_map[pred] - position_map[gt])
    return max(0.0, 1.0 - alpha * d)


def fuzzy_speed_score(pred, gt, alpha=ALPHA_SPEED_DEFAULT):
    return _ordinal_score(pred, gt, SPEED_POSITION, alpha)


def fuzzy_direction_score(pred, gt, alpha=ALPHA_DIRECTION_DEFAULT):
    return _ordinal_score(pred, gt, DIRECTION_POSITION, alpha)


def fuzzy_full_score_min(dir_score, spd_score):
    """Strict fuzzy AND — weakest link dominates."""
    return min(dir_score, spd_score)


def fuzzy_full_score_product(dir_score, spd_score):
    """Multiplicative fuzzy AND — allows partial compensation."""
    return dir_score * spd_score


# ── Field accessors ──────────────────────────────────────────────────
# Handle both shapes that may appear in inference_outputs_v6.json:
#   (a) Cell-3 shape from run_condition():
#       out has 'direction', 'speed', 'gt_direction', 'gt_speed'
#   (b) Cell-6 shape that nests GT inside 'gt_continuous':
#       out['gt_continuous']['direction_label'], etc.

def _get_pred_direction(out):
    return (out.get('direction') or '').strip()

def _get_pred_speed(out):
    return (out.get('speed') or '').strip()

def _get_gt_direction(entry, out):
    # Prefer top-level entry, then condition dict, then gt_continuous
    if entry is not None:
        gt = entry.get('gt_continuous') or {}
        if gt.get('direction_label'):
            return gt['direction_label']
    if out.get('gt_direction'):
        return out['gt_direction']
    return ''

def _get_gt_speed(entry, out):
    if entry is not None:
        gt = entry.get('gt_continuous') or {}
        if gt.get('speed_label'):
            return gt['speed_label']
    if out.get('gt_speed'):
        return out['gt_speed']
    return ''


# ── Score the full results list in place ─────────────────────────────

def score_inference_outputs(results, conditions=('non_posthoc', 'posthoc'),
                             alpha_speed=ALPHA_SPEED_DEFAULT,
                             alpha_direction=ALPHA_DIRECTION_DEFAULT):
    """
    Add fuzzy_dir / fuzzy_spd / fuzzy_full_min / fuzzy_full_prod fields
    to each (entry, condition) pair in `results`. Modifies in place.

    Args:
        results : list of result dicts from inference_outputs_v6.json.
                  Each entry has keys including 'non_posthoc', 'posthoc',
                  and 'gt_continuous'.

    Returns:
        results (modified in place; also returned).
    """
    for entry in results:
        for cond in conditions:
            if cond not in entry:
                continue
            out = entry[cond]

            pred_dir = _get_pred_direction(out)
            pred_spd = _get_pred_speed(out)
            gt_dir   = _get_gt_direction(entry, out)
            gt_spd   = _get_gt_speed(entry, out)

            dscore = fuzzy_direction_score(pred_dir, gt_dir, alpha=alpha_direction)
            sscore = fuzzy_speed_score(pred_spd, gt_spd,     alpha=alpha_speed)

            out['fuzzy_dir']       = dscore
            out['fuzzy_spd']       = sscore
            out['fuzzy_full_min']  = fuzzy_full_score_min(dscore, sscore)
            out['fuzzy_full_prod'] = fuzzy_full_score_product(dscore, sscore)

    return results


# ── Aggregation ──────────────────────────────────────────────────────

def fuzzy_accuracy_summary(results, conditions=('non_posthoc', 'posthoc'),
                            print_table=True):
    """
    Aggregate hard and fuzzy accuracies per condition and print a
    comparison table mirroring the existing summary at end of Cell 6.

    Returns dict of {metric_name: {condition: float}}.
    """
    def hard_dir(entry, out):
        return float(_get_pred_direction(out) == _get_gt_direction(entry, out))

    def hard_spd(entry, out):
        return float(_get_pred_speed(out) == _get_gt_speed(entry, out))

    def hard_full(entry, out):
        return hard_dir(entry, out) * hard_spd(entry, out)

    metrics = {
        'Direction acc (hard)':         hard_dir,
        'Direction acc (fuzzy)':        lambda e, o: o.get('fuzzy_dir', 0.0),
        'Speed acc (hard)':             hard_spd,
        'Speed acc (fuzzy)':            lambda e, o: o.get('fuzzy_spd', 0.0),
        'Full match acc (hard)':        hard_full,
        'Full match acc (fuzzy-min)':   lambda e, o: o.get('fuzzy_full_min', 0.0),
        'Full match acc (fuzzy-prod)':  lambda e, o: o.get('fuzzy_full_prod', 0.0),
    }

    summary = {}
    for name, getter in metrics.items():
        summary[name] = {}
        for cond in conditions:
            total, count = 0.0, 0
            for entry in results:
                if cond not in entry:
                    continue
                total += getter(entry, entry[cond])
                count += 1
            summary[name][cond] = total / count if count else 0.0

    if print_table:
        c1, c2 = conditions
        print(f"\n{'Metric':<32} {c1:>12} {c2:>10} {'Delta':>10}")
        print('-' * 66)
        for name, scores in summary.items():
            v1, v2 = scores.get(c1, 0.0), scores.get(c2, 0.0)
            delta = v2 - v1
            sign  = '+' if delta >= 0 else ''
            print(f"{name:<32} {v1:>12.3f} {v2:>10.3f} {sign}{delta:>9.3f}")

    return summary


# ── Sensitivity analysis ─────────────────────────────────────────────

def alpha_sensitivity_sweep(results, conditions=('non_posthoc', 'posthoc'),
                             alphas=(0.25, 0.33, 0.5, 0.67, 1.0)):
    """
    Re-score with multiple alpha values and print how the metric
    changes. Demonstrates whether conclusions are robust to alpha.
    """
    print(f"\nAlpha sensitivity sweep — Full match (fuzzy-min)")
    print(f"{'alpha':>8} | " + ' | '.join(f"{c:>12}" for c in conditions) + " | Delta")
    print('-' * 60)

    for alpha in alphas:
        per_cond = {c: [] for c in conditions}
        for entry in results:
            for cond in conditions:
                if cond not in entry:
                    continue
                o = entry[cond]
                ds = fuzzy_direction_score(_get_pred_direction(o),
                                            _get_gt_direction(entry, o),
                                            alpha=alpha)
                ss = fuzzy_speed_score(_get_pred_speed(o),
                                        _get_gt_speed(entry, o),
                                        alpha=alpha)
                per_cond[cond].append(min(ds, ss))

        means = {c: sum(v) / len(v) if v else 0.0 for c, v in per_cond.items()}
        c1, c2 = conditions
        delta = means[c2] - means[c1]
        sign  = '+' if delta >= 0 else ''
        row = f"{alpha:>8.2f} | " + ' | '.join(f"{means[c]:>12.3f}" for c in conditions)
        print(f"{row} | {sign}{delta:.3f}")
