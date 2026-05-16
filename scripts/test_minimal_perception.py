"""
Minimal perception test: run only Step 1 with a simple prompt
on one sample, see what Qwen actually describes.
"""
import json, os, gc, sys, socket
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
PROMPT_FILE = "prompts/step1_perception_minimal.txt"
OUTPUT_FILE = "results/perception_minimal_test.json"
FRAME_DT = 0.5  # seconds between consecutive frames — adjust to your dataset

# ── Load data ─────────────────────────────────────────────────────────────────
with open("results/preprocessed_target.json") as f:
    data = json.load(f)
sample = next(s for s in data if s["sample_token"].startswith(SAMPLE_TOKEN_PREFIX))
print(f"Sample: {sample['scene_name']} / {sample['sample_token']}")
print(f"Available fields in sample: {list(sample.keys())}")

prompt_text = open(PROMPT_FILE).read()

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="cuda")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Helpers ───────────────────────────────────────────────────────────────────
def resize(path):
    rp = path.replace(".jpg", "_r.jpg")
    if not os.path.exists(rp):
        Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS).save(rp, quality=85)
    return rp

def build_header(sample, n_past, dt=FRAME_DT):
    """Construct a header string describing ego state and image sequence."""
    gt = sample.get("gt_continuous", {})

    # speed_1_mps is the most recent past speed (closest to t0)
    speed = gt.get("speed_1_mps")
    accel = gt.get("acceleration_mps2")
    goal = sample.get("ego_goal") or gt.get("ego_goal", "continue forward")
    direction = gt.get("direction_label", "")
    speed_label = gt.get("speed_label", "")

    speed_str = f"{speed:.2f} m/s" if speed is not None else "unknown"
    accel_str = f"{accel:.2f} m/s^2" if accel is not None else "unknown"

    motion_descriptor = ""
    if speed_label or direction:
        motion_descriptor = f"Ego motion: {speed_label}, {direction}\n".replace(", ,", ",").strip(", \n") + "\n"

    if n_past == 0:
        image_block = "Single frame from the front camera:\n  image 1: t0 (now)"
    else:
        timesteps = [f"t-{i} ({i*dt:.1f}s ago)" for i in range(n_past, 0, -1)] + ["t0 (now)"]
        lines = [f"  image {i+1}: {label}" for i, label in enumerate(timesteps)]
        image_block = (
            "Image order (continuous sequence from the same front camera, "
            f"{dt}s between frames):\n" + "\n".join(lines)
        )

    return (
        "HEADER\n"
        f"Ego goal: {goal}\n"
        f"Ego speed: {speed_str}\n"
        f"Ego acceleration: {accel_str}\n"
        f"{motion_descriptor}"
        f"{image_block}\n\n"
    )

# ── Build inputs for past + t0 ────────────────────────────────────────────────
fd = sample["frame_dir"]
n_past = sample.get("n_past", 0)
past_imgs = [f"{fd}/tm{i}.jpg" for i in range(n_past, 0, -1)]
imgs = [resize(p) for p in past_imgs + [f"{fd}/t0.jpg"]]
print(f"Using {len(imgs)} images: {len(past_imgs)} past + 1 current")

# ── Inference function ────────────────────────────────────────────────────────
results = {}

def run_inference(images, label, n_past_for_header, max_tokens=400):
    torch.cuda.empty_cache(); gc.collect()
    header = build_header(sample, n_past=n_past_for_header)
    full_prompt = header + prompt_text

    if label == "t0_only":  # print header once for inspection
        print("\n--- HEADER SENT TO MODEL ---")
        print(header)
        print("--- END HEADER ---\n")

    messages = [{"role": "user", "content":
        [{"type": "image", "image": p} for p in images] +
        [{"type": "text", "text": full_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=0.0)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed[0], skip_special_tokens=True)

# ── Run tests ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 1: t0 only (single image, simplest case)")
print("=" * 60)
t0_path = f"{fd}/t0_r.jpg" if os.path.exists(f"{fd}/t0_r.jpg") else resize(f"{fd}/t0.jpg")
out1 = run_inference([t0_path], "t0_only", n_past_for_header=0)
print(out1)
results["t0_only"] = out1

print("\n" + "=" * 60)
print("TEST 2: past + t0 (4 images, same as inference)")
print("=" * 60)
out2 = run_inference(imgs, "past_plus_t0", n_past_for_header=n_past)
print(out2)
results["past_plus_t0"] = out2

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "sample_token": sample["sample_token"],
        "scene_name": sample["scene_name"],
        "prompt": prompt_text,
        "header_example": build_header(sample, n_past=n_past),
        "tests": results,
    }, f, indent=2)
print(f"\nSaved to {OUTPUT_FILE}")
