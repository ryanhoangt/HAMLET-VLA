#!/bin/bash
#SBATCH --job-name=eval_hamlet_libero
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -e

echo "Start at: $(date)"
echo "Running on node: $(hostname)"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

cd /mnt/data/vhoangth2/repos/HAMLET-VLA

export PYTHONNOUSERSITE=1
export WANDB_DISABLE_SERVICE=true
export MUJOCO_GL=osmesa
export QT_QPA_PLATFORM=offscreen

# HAMLET eval runs local inference (no ZMQ server) because the moment_history
# buffer must persist across timesteps and reset between episodes.
PYTHONPATH=/mnt/data/vhoangth2/repos/LIBERO:$PYTHONPATH \
python examples/Libero/eval/run_hamlet_libero_eval.py \
  --base_model_path expdata/libero_joint_finetuning_60k_baseline/checkpoint-60000 \
  --hamlet_checkpoint expdata/hamlet_libero_spatial/hamlet_latest.pt \
  --data_config "examples.Libero.custom_data_config:LiberoDataConfig" \
  --task_suite_name libero_spatial \
  --headless \
  --num_trials_per_task 20

echo "Done at: $(date)"
