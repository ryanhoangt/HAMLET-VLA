#!/bin/bash
#SBATCH --job-name=hamlet_finetuning

#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err

#SBATCH --time=72:00:00

#SBATCH --partition=main

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --gres=gpu:1

echo "Start at: $(date)"
echo "Running on node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

cd /mnt/data/vhoangth2/repos/HAMLET-VLA

export PYTHONNOUSERSITE=1
export WANDB_DISABLE_SERVICE=true
# export WANDB_MODE=offline

export TMPDIR=/home/vhoangth2/pytorch_tmp

# HAMLET finetuning: trains moment_tokens + MemoryModule + action head
# while keeping backbone LLM + visual encoder frozen.
#
# Recommended workflow:
#   1. Run pretrain_moment_tokens.py first to get tcl_latest.pt
#   2. (Optional) run gr00t_finetune.py to adapt the action head to your dataset
#      and save a checkpoint, then pass that checkpoint as --base_model_path
#   3. Run this script with --tcl_checkpoint pointing to tcl_latest.pt
python scripts/hamlet_finetune.py \
    --dataset_path /mnt/data/vhoangth2/hf_cache/lerobot/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --data_config "examples.Libero.custom_data_config:LiberoDataConfig" \
    --base_model_path nvidia/GR00T-N1.5-3B \
    --tcl_checkpoint /mnt/data/vhoangth2/checkpoints/tcl_latest.pt \
    --output_dir expdata/hamlet_libero_spatial \
    --num_gpus 1 \
    --batch_size 8 \
    --max_steps 10000 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --hamlet_lr 1e-4 \
    --warmup_steps 500 \
    --num_workers 8 \
    --prefetch_factor 2 \
    --history_len 4 \
    --history_stride 1 \
    --num_moment_tokens 4 \
    --wandb_project hamlet-libero

echo "Done at: $(date)"
