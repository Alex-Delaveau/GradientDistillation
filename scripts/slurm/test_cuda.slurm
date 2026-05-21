#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --account=rbw@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_cuda_%j.out
#SBATCH --error=logs/test_cuda_%j.err

set -e
module purge
module load pytorch-gpu/py3/2.8.0
source $WORK/GradientDistillation/.venv/bin/activate

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
x = torch.randn(2000, 2000, device='cuda')
y = x @ x.T
print('GPU compute OK, result mean:', y.mean().item())
"