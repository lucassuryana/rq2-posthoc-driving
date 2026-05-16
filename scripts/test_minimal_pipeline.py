"""
Test minimal versions of Step 1 (perception) + Step 2 (comprehension)
on one sample. Sees if the simplified pipeline produces grounded output
through two steps.
"""
import json, os, gc, socket
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Environment ───────────────────────────────────────────────────────────────
hostname = socket.gethostname()
if "daic" in hostname or "tudelft" in hostname or "hpc" in hostname:
    os.environ["HF_HOME"] = "/tudelft.net/staff-umbrella/lsuryana/huggingface"
    MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
else:
    MODEL_PATH = "/home/lsuryana/.cache/huggingface/hub/Qwen2-VL-7B-Instruct"

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_TOKEN_PREFIX = "17e2c5b3"
IMAGE_SIZE = (640, 360)
P1_FILE = "prompts/step1_perception_minimal.txt"
P2_FILE = "prompts/step2_comprehension_minimal.txt"
OUTPUT_FILE = "results/minimal_pipeline_test.json"

# ── Load data ─────────────────────────────────────────────────────────────────
with open("results/preprocessed_target.json") as f:
    data = json.load(f)
sample = next(s for s in data if s["sample_token"].startswith(SAMPLE_TOKEN_PREFIX))
print(f"Sample: {sample['scene_name']} / {sample['sample_token']}")

p1_text = open(P1_FILE).read()
p2_text = open(P2_FILE).read()

# Mini header (same fields the real script builds, simplified)
gt = sample.get("gt_continuous", {})
n_past = sample.get("n_past", 0)
header = f"""You have access to {n_past + 1} images:
- Image 1..{n_past}: past frames at t-{n_past} to t-1 (each 0.5 sec apart)
- Image {n_past + 1}: current frame (t0, decision point)

Ego vehicle goal: {gt.get('ego_goal', 'not specified')}
Ego past motion: avg speed {gt.get('avg_speed_kph', 0)} km/h, acceleration {gt.get('acceleration_mps2', 0)} m/s2, heading {gt.get('angle_overall_deg', 0)} deg
"""

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="cuda")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")


def resize(path):
    rp = path.replace(".jpg", "_r.jpg")
    if not os.path.exists(rp):
        Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS).save(rp, quality=85)
    return rp


def run(messages, max_tokens):
    torch.cuda.empty_cache(); gc.collect()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=0.0)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed[0], skip_special_tokens=True)


# ── Images ────────────────────────────────────────────────────────────────────
fd = sample["frame_dir"]
past_imgs = [f"{fd}/tm{i}.jpg" for i in range(n_past, 0, -1)]
imgs = [resize(p) for p in past_imgs + [f"{fd}/t0.jpg"]]
print(f"Using {len(imgs)} images")

# ── Step 1: Perception ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Perception (minimal prompt)")
print("=" * 60)
prompt1 = header + "\n" + p1_text
msg1 = [{"role": "user", "content":
    [{"type": "image", "image": p} for p in imgs] +
    [{"type": "text", "text": prompt1}]}]
perception_out = run(msg1, max_tokens=400)
print(perception_out)

# ── Step 2: Comprehension ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Comprehension (minimal prompt)")
print("=" * 60)
prompt2 = header + f"\nPerception output:\n{perception_out}\n\n" + p2_text
msg2 = [{"role": "user", "content":
    [{"type": "image", "image": p} for p in imgs] +
    [{"type": "text", "text": prompt2}]}]
comprehension_out = run(msg2, max_tokens=400)
print(comprehension_out)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "sample_token": sample["sample_token"],
        "scene_name": sample["scene_name"],
        "header": header,
        "perception_prompt": p1_text,
        "perception_output": perception_out,
        "comprehension_prompt": p2_text,
        "comprehension_output": comprehension_out,
    }, f, indent=2)
print(f"\nSaved to {OUTPUT_FILE}")

