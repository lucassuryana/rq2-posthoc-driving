#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source ~/.bashrc
conda activate rq2
cd ~/rq2-posthoc-driving
export HF_HOME=/tudelft.net/staff-umbrella/lsuryana/huggingface
jupyter nbconvert --to notebook --execute smoke_test_v6.ipynb --output smoke_test_v6_output.ipynb
