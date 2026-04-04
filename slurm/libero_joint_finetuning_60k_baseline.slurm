#!/bin/bash
#SBATCH --job-name=libero_joint_finetuning_60k_baseline          # Job name shown in squeue/sacct. Keep it short and descriptive (e.g., train_resnet, eval_v1).

#SBATCH --output=log_%j.out          # Stdout log file (normal prints). %j is replaced by the JOB_ID to avoid overwriting logs.

#SBATCH --error=log_%j.err           # Stderr log file (errors/tracebacks). Separating stdout/stderr makes debugging easier.

#SBATCH --time=72:00:00              # Max runtime (HH:MM:SS). When the limit is reached, Slurm will stop the job (TIMEOUT).
                                     # Best practice: test with ~00:10:00 first before running long jobs.
                                     # so you may omit --time (or try --time=0) if you truly need “unlimited”.
                                     
#SBATCH --partition=main             # Which partition/queue to use. Your cluster only has 'main' (and it's the default).
                                     # Keeping it explicit makes the script clearer and more portable.
                                     
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch. 1 = a single main process (common for single-node Python training).
                                     # For MPI / multi-process workloads, this is typically >1.
                                     
#SBATCH --cpus-per-task=8            # CPU threads allocated per task. Increase if data loading / preprocessing is heavy (e.g., 8–32 depending on workload).

#SBATCH --mem=64G                    # Memory request for the job.
                                     # Too low => OOM; too high => wasted resources / longer queue time.
                                     
#SBATCH --gres=gpu:1                 # Request GPUs via GRES (Generic RESource). gpu:1 = request 1 GPU. Remove this line for CPU-only jobs.
                                     # On your compute nodes you have H100 80GB GPUs; request gpu:2/4/8 for multi-GPU training (plus DDP setup).
                                     # Slurm usually sets CUDA_VISIBLE_DEVICES automatically when GPUs are allocated.

# --- Recommended extras (optional) ---
# 1) Store large data and outputs under /mnt/data
# 2) If using conda inside the job:
#    source $HOME/miniconda3/etc/profile.d/conda.sh
#    conda activate <env_name>
#
# If you need to install Miniconda, see the guide below.

echo "Start at: $(date)"
echo "Running on node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

cd /mnt/data/vhoangth2/repos/HAMLET-VLA

export PYTHONNOUSERSITE=1
export WANDB_DISABLE_SERVICE=true
# export WANDB_MODE=offline
# export WANDB_DISABLE_CODE=true

export TMPDIR=/home/vhoangth2/pytorch_tmp

# Train
python scripts/gr00t_finetune.py --dataset-path /mnt/data/vhoangth2/hf_cache/lerobot/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot /mnt/data/vhoangth2/hf_cache/lerobot/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot /mnt/data/vhoangth2/hf_cache/lerobot/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot /mnt/data/vhoangth2/hf_cache/lerobot/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot --data_config examples.Libero.custom_data_config:LiberoDataConfig --num-gpus 1 --save_steps 5000 --batch-size 128 --output-dir expdata/libero_joint_finetuning_60k_baseline --max-steps 60000 --dataloader_num_workers 8 --dataloader_prefetch_factor 2