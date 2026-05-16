#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source ~/.bashrc
conda activate rq2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/rq2-posthoc-driving
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_HOME=/tudelft.net/staff-umbrella/lsuryana/huggingface

python scripts/03_inference_v6_single.py
