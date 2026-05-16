"""
scripts/03_inference_v6_target.py

V6 5-step inference (SA-structured) for target scenes — converted from the
smoke-test notebook into a standalone script. No plotting, no notebook
widgets, otherwise identical logic.

5 steps per condition:
  Step 1  SA Level 1 — Perception     (images + header)
  Step 2  SA Level 2 — Comprehension  (images + Step 1 Q&A + summary)
  Step 3  SA Level 3 — Projection     (images + Step 2 Q&A + summary)
  Step 4  Planning                    (Step 3 Q&A + summary)            text only
  Step 5  Behavior                    (all summaries + planning)        text only

Two conditions per sample:
  non_posthoc : past frames + t0
  posthoc     : past frames + t0 + future frames + GT future motion in header

Reads:  results/preprocessed_target_single.json
Writes: results/inference_outputs_v6_single.json
        results/inference_checkpoint_v6_single.json (saved after each sample)
"""
import json, os, re, math, gc, time, socket
from collections import Counter

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import continuous_to_drivelm_label
from fuzzy_labels import build_fuzzy_labeler, compare_fuzzy_vs_hard

# ── Environment detection ────────────────────────────────────────────────────
hostname = socket.gethostname()
if "daic" in hostname or "tudelft" in hostname or "hpc" in hostname:
    os.environ["HF_HOME"] = "/tudelft.net/staff-umbrella/lsuryana/huggingface"
    MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
else:
    MODEL_PATH = "/home/lsuryana/.cache/huggingface/hub/Qwen2-VL-7B-Instruct"

print(f"Running on: {hostname}")
print(f"MODEL_PATH:  {MODEL_PATH}")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE        = 'results/preprocessed_target_single.json'
CHECKPOINT_FILE   = 'results/inference_checkpoint_v6_single.json'
OUTPUT_FILE       = 'results/inference_outputs_v6_single.json'
FORCE_FRESH_START = True   # set True to ignore existing checkpoint

IMAGE_SIZE = (640, 360)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="cuda")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print(f"Model loaded. VRAM: {round(torch.cuda.memory_allocated()/1e9,2)} GB")

# ── Load data ─────────────────────────────────────────────────────────────────
with open(INPUT_FILE) as f:
    data = json.load(f)

# Fuzzy Layer 1 labeler (must come from the same dataset distribution)
FUZZY_LABELER = build_fuzzy_labeler(data, verbose=True)
compare_fuzzy_vs_hard(data, FUZZY_LABELER)

# ── Prompts (5 files) ─────────────────────────────────────────────────────────
p1 = open('prompts/step1_perception.txt').read()
p2 = open('prompts/step2_comprehension.txt').read()
p3 = open('prompts/step3_projection.txt').read()
p4 = open('prompts/step4_planning.txt').read()
p5 = open('prompts/step5_behavior.txt').read()

# ── Image helpers ─────────────────────────────────────────────────────────────
def resize_image(path):
    rp = path.replace('.jpg', '_r.jpg')
    if not os.path.exists(rp):
        Image.open(path).convert('RGB').resize(IMAGE_SIZE, Image.LANCZOS)\
             .save(rp, quality=85)
    return rp


def get_image_sequence(s, condition):
    fd     = s['frame_dir']
    n_past = s.get('n_past', 0)
    past_frames = []
    if n_past >= 3:
        past_frames = [f"{fd}/tm3.jpg", f"{fd}/tm2.jpg", f"{fd}/tm1.jpg"]
    elif n_past == 2:
        past_frames = [f"{fd}/tm2.jpg", f"{fd}/tm1.jpg"]
    elif n_past == 1:
        past_frames = [f"{fd}/tm1.jpg"]
    future_frames = [f"{fd}/t1.jpg", f"{fd}/t2.jpg", f"{fd}/t3.jpg"] \
                    if condition == 'posthoc' else []
    return past_frames + [f"{fd}/t0.jpg"] + future_frames


def build_image_description(s, condition):
    n_past = s.get('n_past', 0)
    lines  = []
    if n_past == 3:
        lines = [
            "- Image 1 (tm3_r.jpg): t-3 (1.5 seconds before decision point)",
            "- Image 2 (tm2_r.jpg): t-2 (1.0 second before decision point)",
            "- Image 3 (tm1_r.jpg): t-1 (0.5 seconds before decision point)",
            "- Image 4 (t0_r.jpg): t0  (current frame -- DECISION POINT)",
        ]
    elif n_past == 2:
        lines = [
            "- Image 1 (tm2_r.jpg): t-2 (1.0 second before decision point)",
            "- Image 2 (tm1_r.jpg): t-1 (0.5 seconds before decision point)",
            "- Image 3 (tm0_r.jpg): t0  (current frame -- DECISION POINT)",
        ]
    elif n_past == 1:
        lines = [
            "- Image 1 (tm1_r.jpg): t-1 (0.5 seconds before decision point)",
            "- Image 2 (t0_r.jpg): t0  (current frame -- DECISION POINT)",
        ]
    else:
        lines = ["- Image 1: t0 (current frame -- DECISION POINT)"]
    if condition == 'posthoc':
        offset = n_past + 1
        lines += [
            f"- Image {offset+0} (t1_r.jpg): t+1 (0.5 seconds after decision point)",
            f"- Image {offset+1} (t2_r.jpg): t+2 (1.0 second after decision point)",
            f"- Image {offset+2} (t3_r.jpg): t+3 (1.5 seconds after decision point)",
        ]
    return "\n".join(lines)

# ── Inference helpers ─────────────────────────────────────────────────────────
def run_inference(messages, max_tokens=128):
    torch.cuda.empty_cache()
    gc.collect()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    has_images = any(
        isinstance(item, dict) and item.get('type') == 'image'
        for msg in messages for item in msg.get('content', [])
    )
    if has_images:
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs,
                           return_tensors="pt").to("cuda")
    else:
        inputs = processor(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.0, do_sample=False)
    trimmed = out[:, inputs['input_ids'].shape[1]:]
    result  = processor.decode(trimmed[0], skip_special_tokens=True)
    del inputs, out, trimmed
    torch.cuda.empty_cache()
    return result


def parse_step(raw, field):
    patterns = {
        'perception_summary':    r'PERCEPTION_SUMMARY:\s*(.*?)$',
        'comprehension_summary': r'COMPREHENSION_SUMMARY:\s*(.*?)$',
        'prediction_summary':    r'PREDICTION_SUMMARY:\s*(.*?)$',
        'planning':              r'PLANNING:\s*(.*?)$',
        'direction':             r'DIRECTION:\s*(.+?)(?:\n|$)',
        'speed':                 r'SPEED:\s*(.+?)(?:\n|$)',
        'reasoning':             r'A16:\s*(.*?)$',
    }
    m = re.search(patterns[field], raw, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else ''

# ── Data access helpers ───────────────────────────────────────────────────────
def get_gt_continuous(s):
    return s.get('gt_continuous') or {}


def get_past_behavior_label(past_traj, n_past):
    if n_past == 0 or not past_traj:
        return FUZZY_LABELER(0.0, 0.0)
    earliest      = past_traj[0]
    dx            = 0.0 - earliest['x']
    dy            = 0.0 - earliest['y']
    dist          = math.sqrt(dx**2 + dy**2)
    avg_speed_mps = dist / (n_past * 0.5)
    angle_deg     = math.degrees(math.atan2(dy, dx)) if dist > 0.01 else 0.0
    return FUZZY_LABELER(avg_speed_mps, angle_deg)


def compute_past_motion_state(past_traj, n_past):
    if n_past == 0 or not past_traj:
        return None
    points = past_traj + [{'x': 0.0, 'y': 0.0}]
    interval_speeds = []
    for i in range(len(points) - 1):
        dx = points[i+1]['x'] - points[i]['x']
        dy = points[i+1]['y'] - points[i]['y']
        interval_speeds.append(math.sqrt(dx**2 + dy**2) / 0.5)
    avg_speed_mps = round(sum(interval_speeds) / len(interval_speeds), 2)
    avg_speed_kph = round(avg_speed_mps * 3.6, 1)
    if len(interval_speeds) >= 2:
        dv = interval_speeds[-1] - interval_speeds[0]
        dt = (len(interval_speeds) - 1) * 0.5
        acceleration_mps2 = round(dv / dt, 3)
    else:
        acceleration_mps2 = None
    earliest   = past_traj[0]
    dx_total   = 0.0 - earliest['x']
    dy_total   = 0.0 - earliest['y']
    dist_total = math.sqrt(dx_total**2 + dy_total**2)
    heading_deg = round(math.degrees(math.atan2(dy_total, dx_total)), 2) \
                  if dist_total > 0.01 else 0.0
    yaw_rate_deg_s = None
    if n_past >= 3 and len(points) >= 3:
        headings = []
        for i in range(len(points) - 1):
            dx = points[i+1]['x'] - points[i]['x']
            dy = points[i+1]['y'] - points[i]['y']
            if math.sqrt(dx**2 + dy**2) > 0.01:
                headings.append(math.degrees(math.atan2(dy, dx)))
        if len(headings) >= 2:
            deltas = [(((headings[i+1] - headings[i]) + 180) % 360 - 180) / 0.5
                      for i in range(len(headings) - 1)]
            yaw_rate_deg_s = round(sum(deltas) / len(deltas), 2)
    return {
        'avg_speed_mps':     avg_speed_mps,
        'avg_speed_kph':     avg_speed_kph,
        'acceleration_mps2': acceleration_mps2,
        'heading_deg':       heading_deg,
        'yaw_rate_deg_s':    yaw_rate_deg_s,
    }


def build_location_line(gt_cont):
    city       = gt_cont.get('city', '')
    country    = gt_cont.get('country', '')
    drive_side = gt_cont.get('drive_side', '')
    if not city or country == 'unknown':
        return ''
    return (f"Recording location: {city}, {country} "
            f"(drive on the {drive_side} side of the road)\n")


def build_context_header(s, condition):
    n_past    = s.get('n_past', 0)
    traj      = s['trajectory']
    past_traj = s.get('past_trajectory', [])
    gt_cont   = get_gt_continuous(s)
    ego_goal  = gt_cont.get('ego_goal', 'not specified')

    img_desc = build_image_description(s, condition)
    n_total  = len(get_image_sequence(s, condition))

    if n_past >= 1:
        dir_lbl, spd_lbl   = get_past_behavior_label(past_traj, n_past)
        past_behavior_text = f"Ego vehicle past behavior: {dir_lbl}, {spd_lbl}\n"
    else:
        past_behavior_text = "Ego vehicle past behavior: not available\n"

    motion = compute_past_motion_state(past_traj, n_past)
    if motion:
        motion_text = (
            f"Ego vehicle past motion state:\n"
            f"  avg speed:    {motion['avg_speed_mps']} m/s "
            f"({motion['avg_speed_kph']} km/h)\n"
            f"  acceleration: {motion['acceleration_mps2']} m/s2\n"
            f"  heading:      {motion['heading_deg']} degrees\n"
        )
        if motion['yaw_rate_deg_s'] is not None:
            motion_text += f"  yaw rate:     {motion['yaw_rate_deg_s']} deg/s\n"
    else:
        motion_text = "Ego vehicle past motion state: not available\n"

    location_line = build_location_line(gt_cont)

    header = (
        f"You have access to {n_total} images showing the driving scene:\n"
        f"{img_desc}\n\n"
        f"{location_line}"
        f"Ego vehicle goal: {ego_goal}\n\n"
        f"{past_behavior_text}"
        f"{motion_text}"
    )

    if condition == 'posthoc':
        header += (
            f"\nEgo vehicle future trajectory (ground-truth):\n"
            f"  t+0.5s: x={traj[0]['x']}, y={traj[0]['y']}\n"
            f"  t+1.0s: x={traj[1]['x']}, y={traj[1]['y']}\n"
            f"  t+1.5s: x={traj[2]['x']}, y={traj[2]['y']}\n"
            f"Ego vehicle future motion state (ground-truth):\n"
            f"  avg speed:    {gt_cont.get('avg_speed_mps', 0)} m/s "
            f"({gt_cont.get('avg_speed_kph', 0)} km/h)\n"
            f"  heading:      {gt_cont.get('angle_overall_deg', 0)} degrees\n"
            f"  acceleration: {gt_cont.get('acceleration_mps2', 0)} m/s2\n"
            f"  yaw rate:     {gt_cont.get('yaw_rate_deg_s', 0)} deg/s\n"
        )
    return header


def run_condition(s, condition):
    resized        = [resize_image(p) for p in get_image_sequence(s, condition)]
    context_header = build_context_header(s, condition)

    def image_messages():
        return [{"type": "image", "image": p} for p in resized]

    # Step 1: Perception
    prompt1 = context_header + "\n" + p1
    msg1    = [{"role": "user", "content":
                image_messages() + [{"type": "text", "text": prompt1}]}]
    raw1    = run_inference(msg1, max_tokens=1200)
    perc_summary = parse_step(raw1, 'perception_summary') or raw1.strip()

    # Step 2: Comprehension
    prompt2 = (context_header
               + f"\nPerception output:\n{raw1}\nPerception summary: {perc_summary}\n\n"
               + p2)
    msg2    = [{"role": "user", "content":
                image_messages() + [{"type": "text", "text": prompt2}]}]
    raw2    = run_inference(msg2, max_tokens=896)
    comp_summary = parse_step(raw2, 'comprehension_summary') or raw2.strip()

    # Step 3: Projection
    prompt3 = (context_header
               + f"\nComprehension output:\n{raw2}\nComprehension summary: {comp_summary}\n\n"
               + p3)
    msg3    = [{"role": "user", "content":
                image_messages() + [{"type": "text", "text": prompt3}]}]
    raw3    = run_inference(msg3, max_tokens=768)
    pred_summary = parse_step(raw3, 'prediction_summary') or raw3.strip()

    # Step 4: Planning (text only)
    prompt4 = (context_header
               + f"\nProjection output:\n{raw3}\nProjection summary: {pred_summary}\n\n"
               + p4)
    msg4    = [{"role": "user", "content": [{"type": "text", "text": prompt4}]}]
    raw4    = run_inference(msg4, max_tokens=512)
    planning = parse_step(raw4, 'planning') or raw4.strip()

    # Step 5: Behavior (text only)
    prompt5 = (context_header
               + f"\nPerception summary:    {perc_summary}\n"
               + f"Comprehension summary: {comp_summary}\n"
               + f"Prediction summary:    {pred_summary}\n"
               + f"Planning recommendation: {planning}\n\n"
               + p5)
    msg5    = [{"role": "user", "content": [{"type": "text", "text": prompt5}]}]
    raw5    = run_inference(msg5, max_tokens=256)
    direction = parse_step(raw5, 'direction')
    speed     = parse_step(raw5, 'speed')
    reasoning = parse_step(raw5, 'reasoning')

    gt = get_gt_continuous(s)
    gt_direction = gt.get('direction_label', '')
    gt_speed     = gt.get('speed_label', '')

    return {
        'context_header':        context_header,
        'perception_raw':        raw1,
        'perception_summary':    perc_summary,
        'comprehension_raw':     raw2,
        'comprehension_summary': comp_summary,
        'prediction_raw':        raw3,
        'prediction_summary':    pred_summary,
        'planning':              planning,
        'direction':             direction,
        'speed':                 speed,
        'reasoning':             reasoning,
        'raw_step5':             raw5,
        'gt_direction':          gt_direction,
        'gt_speed':              gt_speed,
        'direction_correct':     direction == gt_direction,
        'speed_correct':         speed == gt_speed,
        'full_correct':          direction == gt_direction and speed == gt_speed,
        'n_images':              len(resized),
        'ego_avg_speed_mps':     gt.get('avg_speed_mps', 0),
        'ego_avg_speed_kph':     gt.get('avg_speed_kph', 0),
        'ego_acceleration_mps2': gt.get('acceleration_mps2', 0),
        'ego_heading_deg':       gt.get('angle_overall_deg', 0),
    }

# ── Checkpoint / resume ───────────────────────────────────────────────────────
results     = []
done_tokens = set()

if FORCE_FRESH_START and os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Checkpoint deleted -- starting fresh run.")
elif os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        results = json.load(f)
    done_tokens = {r['sample_token'] for r in results}
    print(f"Resuming from checkpoint: {len(done_tokens)}/{len(data)} done")
else:
    print("No checkpoint -- starting fresh run.")

# ── Main loop ─────────────────────────────────────────────────────────────────
total_start = time.time()
N = len(data)

for idx, s in enumerate(data):
    if s['sample_token'] in done_tokens:
        print(f"[{idx+1}/{N}] Skipping {s['sample_token'][:8]} (done)")
        continue

    gt     = get_gt_continuous(s)
    n_past = s.get('n_past', 0)
    print(f"\n[{idx+1}/{N}] {s.get('scene_name','?')} {s['sample_token'][:8]}...  "
          f"GT: {gt.get('avg_speed_kph')} km/h, "
          f"{gt.get('angle_overall_deg')} deg")

    print(f"  Non-posthoc...", end=' ')
    t_start = time.time()
    out_non = run_condition(s, 'non_posthoc')
    print(f"{time.time()-t_start:.1f}s  ->  "
          f"dir={'OK' if out_non['direction_correct'] else 'WRONG'} ({out_non['direction']}), "
          f"spd={'OK' if out_non['speed_correct'] else 'WRONG'} ({out_non['speed']})")

    print(f"  Posthoc...    ", end=' ')
    t_start = time.time()
    out_post = run_condition(s, 'posthoc')
    print(f"{time.time()-t_start:.1f}s  ->  "
          f"dir={'OK' if out_post['direction_correct'] else 'WRONG'} ({out_post['direction']}), "
          f"spd={'OK' if out_post['speed_correct'] else 'WRONG'} ({out_post['speed']})")

    results.append({
        'sample_token':    s['sample_token'],
        'scene_token':     s['scene_token'],
        'scene_name':      s.get('scene_name', ''),
        'gt_continuous':   gt,
        'trajectory':      s['trajectory'],
        'past_trajectory': s.get('past_trajectory', []),
        'n_past':          n_past,
        'non_posthoc':     out_non,
        'posthoc':         out_post,
    })
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

# ── Final save + summary ──────────────────────────────────────────────────────
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

n = len(results)
total_time = time.time() - total_start
dir_non   = sum(r['non_posthoc']['direction_correct'] for r in results) / n
dir_post  = sum(r['posthoc']['direction_correct']     for r in results) / n
spd_non   = sum(r['non_posthoc']['speed_correct']     for r in results) / n
spd_post  = sum(r['posthoc']['speed_correct']         for r in results) / n
full_non  = sum(r['non_posthoc']['full_correct']      for r in results) / n
full_post = sum(r['posthoc']['full_correct']          for r in results) / n

print(f"\n{'='*55}")
print(f"DONE -- {n} samples in {total_time/60:.1f} minutes")
print(f"\n{'Metric':<20} {'Non-posthoc':>14} {'Posthoc':>10} {'Delta':>10}")
print("-" * 55)
print(f"{'Direction acc':<20} {dir_non:>14.3f} {dir_post:>10.3f} {dir_post - dir_non:>+10.3f}")
print(f"{'Speed acc':<20} {spd_non:>14.3f} {spd_post:>10.3f} {spd_post - spd_non:>+10.3f}")
print(f"{'Full match acc':<20} {full_non:>14.3f} {full_post:>10.3f} {full_post - full_non:>+10.3f}")
print(f"\nSaved to: {OUTPUT_FILE}")
