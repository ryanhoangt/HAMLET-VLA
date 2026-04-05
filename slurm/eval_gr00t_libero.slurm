#!/bin/bash
#SBATCH --job-name=eval_gr00t_libero          # Job name shown in squeue/sacct. Keep it short and descriptive (e.g., train_resnet, eval_v1).

#SBATCH --output=log_%j.out          # Stdout log file (normal prints). %j is replaced by the JOB_ID to avoid overwriting logs.

#SBATCH --error=log_%j.err           # Stderr log file (errors/tracebacks). Separating stdout/stderr makes debugging easier.

#SBATCH --time=48:00:00              # Max runtime (HH:MM:SS). When the limit is reached, Slurm will stop the job (TIMEOUT).
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

# -----------------------------
# 1. Start the inference server
# -----------------------------
python scripts/inference_service.py \
  --model_path expdata/libero_joint_finetuning_60k_baseline/checkpoint-60000 \
  --server \
  --data_config examples.Libero.custom_data_config:LiberoDataConfig \
  --denoising-steps 8 \
  --embodiment-tag new_embodiment \
  --host 0.0.0.0 \
  2>&1 &

SERVER_PID=$!
echo "Started server with PID $SERVER_PID"

# Give the server a few seconds to start up
sleep 20

# -----------------------------
# 2. Run evaluation
# -----------------------------
echo "Running evaluation..."
export MUJOCO_GL=egl
export QT_QPA_PLATFORM=offscreen
cd examples/Libero/eval/

PYTHONPATH=/mnt/data/vhoangth2/repos/LIBERO:$PYTHONPATH \
python run_libero_eval.py \
  --task_suite_name libero_spatial \
  --headless --num_trials_per_task 20

# -----------------------------
# 3. Cleanup
# -----------------------------
echo "Stopping inference server..."
kill $SERVER_PID || true

echo "Done."