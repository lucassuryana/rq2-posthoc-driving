"""
scripts/03_inference.py
Sequential 4-step inference for all 24 samples under both conditions.

Conditions:
  Non post-hoc: t0 only for all 4 steps
  Post-hoc:     t0+t1+t2+t3 for all 4 steps

Steps per sample per condition:
  1. Perception
  2. Prediction  (builds on perception answer)
  3. Planning    (builds on perception + prediction answers)
  4. Behavior    (builds on all previous, outputs final action)

Output: results/inference_outputs_v2.json
"""
import json, os, re, torch, time, gc
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "/home/lsuryana/.cache/huggingface/hub/Qwen2-VL-7B-Instruct"
CHECKPOINT    = "results/inference_checkpoint_v2.json"
OUTPUT_FILE   = "results/inference_outputs_v2.json"
IMAGE_SIZE    = (640, 360)   # resize from 1600x900 to fit in VRAM
MAX_NEW_TOKENS = 256
VALID_ACTIONS = {'KEEP_LANE','ACCELERATE','DECELERATE',
                 'CHANGE_LANE_LEFT','CHANGE_LANE_RIGHT','STOP'}
CHOICE_MAP    = {'A':'KEEP_LANE','B':'ACCELERATE','C':'DECELERATE',
                 'D':'CHANGE_LANE_LEFT','E':'CHANGE_LANE_RIGHT','F':'STOP'}

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="cuda")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print(f"Model loaded. VRAM: {round(torch.cuda.memory_allocated()/1e9,2)} GB")

# ── Load data ─────────────────────────────────────────────────────────────────
with open('results/preprocessed.json') as f:
    data = json.load(f)

# ── Load prompt templates ─────────────────────────────────────────────────────
p1 = open('prompts/step1_perception.txt').read()
p2 = open('prompts/step2_prediction.txt').read()
p3 = open('prompts/step3_planning.txt').read()
p4 = open('prompts/step4_behavior.txt').read()

# ── Helper functions ──────────────────────────────────────────────────────────
def resize_image(path):
    """Resize image to 640x360 and save to temp path."""
    tmp = path.replace('.jpg', '_r.jpg')
    if not os.path.exists(tmp):
        img = Image.open(path).convert('RGB').resize(IMAGE_SIZE, Image.LANCZOS)
        img.save(tmp, quality=85)
    return tmp

def run_inference(messages):
    """Run single inference call, free VRAM after."""
    torch.cuda.empty_cache()
    gc.collect()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0, do_sample=False)
    trimmed = out[:, inputs['input_ids'].shape[1]:]
    result = processor.decode(trimmed[0], skip_special_tokens=True)
    del inputs, out, trimmed
    torch.cuda.empty_cache()
    return result

def parse_step(raw, step):
    """Extract the answer for a given step from raw model output."""
    patterns = {
        'perception': r'PERCEPTION:\s*(.*?)$',
        'prediction': r'PREDICTION:\s*(.*?)$',
        'planning':   r'PLANNING:\s*(.*?)$',
        'choice':     r'CHOICE:\s*([A-F])',
        'action':     r'ACTION:\s*(\w+)',
        'reasoning':  r'REASONING:\s*(.*?)$',
    }
    match = re.search(patterns[step], raw, re.DOTALL)
    return match.group(1).strip() if match else ''

def derive_gt_action(traj):
    """Derive GT action from ego trajectory using heuristic thresholds."""
    x1, x3, y3 = traj[0]['x'], traj[2]['x'], traj[2]['y']
    if abs(x3) < 0.5:   return 'STOP'
    if y3 > 3.0:        return 'CHANGE_LANE_LEFT'
    if y3 < -3.0:       return 'CHANGE_LANE_RIGHT'
    if x1 < 1.5:        return 'DECELERATE'
    if x3 > 12.0:       return 'ACCELERATE'
    return 'KEEP_LANE'

def run_condition(fd, images):
    """
    Run all 4 steps for one condition.
    images: list of image paths [t0] or [t0,t1,t2,t3]
    Returns dict with perception, prediction, planning, action, reasoning
    """
    # Resize all images
    resized = [resize_image(img) for img in images]

    def make_image_content():
        return [{"type": "image", "image": p} for p in resized]

    # Step 1: Perception
    msg1 = [{"role": "user", "content":
        make_image_content() + [{"type": "text", "text": p1}]}]
    raw1 = run_inference(msg1)
    ans_perception = parse_step(raw1, 'perception') or raw1.strip()

    # Step 2: Prediction (includes perception answer)
    prompt2 = p2.replace('{PERCEPTION_ANSWER}', ans_perception)
    msg2 = [{"role": "user", "content":
        make_image_content() + [{"type": "text", "text": prompt2}]}]
    raw2 = run_inference(msg2)
    ans_prediction = parse_step(raw2, 'prediction') or raw2.strip()

    # Step 3: Planning (includes perception + prediction answers)
    prompt3 = p3.replace('{PERCEPTION_ANSWER}', ans_perception)\
                 .replace('{PREDICTION_ANSWER}', ans_prediction)
    msg3 = [{"role": "user", "content":
        make_image_content() + [{"type": "text", "text": prompt3}]}]
    raw3 = run_inference(msg3)
    ans_planning = parse_step(raw3, 'planning') or raw3.strip()

    # Step 4: Behavior (includes all previous answers)
    prompt4 = p4.replace('{PERCEPTION_ANSWER}', ans_perception)\
                 .replace('{PREDICTION_ANSWER}', ans_prediction)\
                 .replace('{PLANNING_ANSWER}', ans_planning)
    msg4 = [{"role": "user", "content":
        make_image_content() + [{"type": "text", "text": prompt4}]}]
    raw4 = run_inference(msg4)

    choice    = parse_step(raw4, 'choice')
    action    = CHOICE_MAP.get(choice, 'PARSE_ERROR')
    reasoning = parse_step(raw4, 'reasoning') or raw4.strip()

    return {
        'perception': ans_perception,
        'prediction': ans_prediction,
        'planning':   ans_planning,
        'action':     action,
        'choice':     choice,
        'reasoning':  reasoning,
    }

# ── Load checkpoint ───────────────────────────────────────────────────────────
done_tokens = set()
results = []
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        results = json.load(f)
    done_tokens = {r['sample_token'] for r in results}
    print(f"Resuming from checkpoint: {len(done_tokens)} done")

# ── Main loop ─────────────────────────────────────────────────────────────────
total_start = time.time()

for idx, s in enumerate(data):
    if s['sample_token'] in done_tokens:
        continue

    fd   = s['frame_dir']
    traj = s['trajectory']
    gt   = derive_gt_action(traj)

    print(f"\n[{idx+1}/24] {s['sample_token'][:8]}... GT={gt}")

    # Non post-hoc: t0 only
    print("  Running non post-hoc (t0 only)...")
    t_start = time.time()
    out_non = run_condition(fd, [f"{fd}/t0.jpg"])
    print(f"  Done in {time.time()-t_start:.1f}s → ACTION: {out_non['action']}")

    # Post-hoc: t0+t1+t2+t3
    print("  Running post-hoc (t0+t1+t2+t3)...")
    t_start = time.time()
    out_post = run_condition(fd, [f"{fd}/t0.jpg", f"{fd}/t1.jpg",
                                   f"{fd}/t2.jpg", f"{fd}/t3.jpg"])
    print(f"  Done in {time.time()-t_start:.1f}s → ACTION: {out_post['action']}")

    result = {
        'sample_token': s['sample_token'],
        'scene_token':  s['scene_token'],
        'gt_action':    gt,
        'trajectory':   traj,
        'non_posthoc':  out_non,
        'posthoc':      out_post,
    }
    results.append(result)

    # Save checkpoint
    with open(CHECKPOINT, 'w') as f:
        json.dump(results, f, indent=2)

# ── Save final output ─────────────────────────────────────────────────────────
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

total_time = time.time() - total_start
change_rate = sum(1 for r in results
    if r['non_posthoc']['action'] != r['posthoc']['action']) / len(results)
acc_non  = sum(1 for r in results
    if r['non_posthoc']['action'] == r['gt_action']) / len(results)
acc_post = sum(1 for r in results
    if r['posthoc']['action'] == r['gt_action']) / len(results)

print(f"\n{'='*50}")
print(f"DONE — {len(results)} samples in {total_time/60:.1f} minutes")
print(f"Non post-hoc accuracy : {acc_non:.3f}")
print(f"Post-hoc accuracy     : {acc_post:.3f}")
print(f"Change rate           : {change_rate:.3f}")
print(f"Saved to              : {OUTPUT_FILE}")
