# RQ2: Post-hoc vs Non-post-hoc Reasoning in Autonomous Driving

## Research Question
How does access to ground-truth future observations influence the reasoning processes and resulting decisions of vision-language models in tactical driving scenarios?

## Setup
```bash
conda create -n rq2 python=3.10 -y
conda activate rq2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0 accelerate qwen-vl-utils nuscenes-devkit sentence-transformers pandas numpy matplotlib seaborn Pillow tqdm pyquaternion
```

## Data
- nuScenes-mini (download from nuscenes.org)
- DriveLM val annotations (HuggingFace: OpenDriveLab/DriveLM → v1_1_val_nus_q_only.json)

## Pipeline
| Script | Description |
|--------|-------------|
| `scripts/01_select_samples.py` | Select overlapping samples |
| `scripts/02_preprocess.py` | Extract frames and ego trajectories |
| `scripts/03_inference.py` | Run Qwen2-VL under both conditions |
| `scripts/04_eval_decision.py` | Decision accuracy vs GT trajectory |
| `scripts/05_eval_reasoning.py` | Change rate + semantic shift |

## Model
Qwen2-VL-7B-Instruct

## Status
- [x] Day 1: Environment setup
- [x] Day 2: Data download (24 samples overlap)
- [x] Day 3: Sample selection
- [x] Day 4: Preprocessing
- [ ] Day 5: VLM smoke test
- [ ] Day 6-7: Full inference
- [ ] Day 8-9: Evaluation
- [ ] Day 10-13: Analysis and write-up
