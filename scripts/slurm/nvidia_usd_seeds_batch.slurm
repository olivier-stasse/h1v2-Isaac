#!/bin/bash
#SBATCH --job-name=batch_h1-nvidia        # Job name
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --ntasks-per-node=1               # One task per node
#SBATCH --gres=gpu:1                      # 1 GPU per job
#SBATCH --cpus-per-task=10                # 10 CPU cores per task
#SBATCH --hint=nomultithread              # Physical cores only
#SBATCH --time=5:00:00                    # Maximum runtime (5 hours)
#SBATCH --output=logs/%x_%A_%a.out        # Output log
#SBATCH --error=logs/%x_%A_%a.err         # Error log
#SBATCH --array=0-4                       # Array job with 5 tasks

export GIT_PYTHON_REFRESH=quiet

# Activate conda environment
module purge
module load miniforge
conda activate /lustre/fswork/projects/rech/mav/uoa29gx/env_isaaclab
set -x

# Define imitation weight variations from 0.0 to 10.0
declare -A CONFIGURATIONS=(
    ["0"]="agent.seed=0"
    ["1"]="agent.seed=10"
    ["2"]="agent.seed=20"
    ["3"]="agent.seed=30"
    ["4"]="agent.seed=40"
)

# Get current configuration
CONFIG="${CONFIGURATIONS[$SLURM_ARRAY_TASK_ID]}"

# Create experiment name with array ID for uniqueness
EXPERIMENT_NAME="h1_nvidia_${SLURM_ARRAY_TASK_ID}"

# CUDA settings
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/pub/anaconda-py3/2023.03/lib/
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1

# Run training with fixed best parameters and varying imitation weight
cd /lustre/fswork/projects/rech/mav/uoa29gx/projects/dog2robot/IsaacLab
python scripts/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-H1-NVIDIA-USD \
    --headless \
    --num_envs 4096 \
    agent.max_iterations=1000 \
    agent.experiment_name="${EXPERIMENT_NAME}" \
    ${CONFIG}

conda deactivate
echo "Job ${SLURM_ARRAY_TASK_ID} completed successfully"
