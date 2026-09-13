#!/bin/bash
#SBATCH -p Abaki
#SBATCH --qos=abaki
#SBATCH --job-name=tc70_nr_ep80
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=24:00:00

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"

cd "$REPO_ROOT"

mkdir -p slurm/logs
mkdir -p "$RUNS_ROOT"

echo "Repository root: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "Runs root: $RUNS_ROOT"

source "$REPO_ROOT/.venv/bin/activate"
echo "Host: $(hostname)"
echo "Start: $(date)"

which python
python --version
nvidia-smi || true

python -m scripts.train \
  --exp-name tc70_nr_ep80 \
  --data-root "$DATA_ROOT" \
  --runs-root "$RUNS_ROOT" \
  --device cuda \
  --seed 18 \
  --num-workers 4 \
  --save-ckpt-every 10 \
  --horizontal-flip \
  --color-jitter \
  --augmentation-strategy baseline \
  --epochs 80 \
  --batch-size 16 \
  --lr-backbone 5e-5 \
  --lr-head 1e-3 \
  --weight-decay 0.05 \
  --scheduler poly_warmup \
  --warmup-epochs 5 \
  --backbone-model timm/vit_small_patch16_224.augreg_in21k_ft_in1k \
  --pretrained \
  --img-size 448 \
  --head-type linear \
  --drop-path 0.05 \
  --dropout 0.0 \
  --att-dropout 0.0 

echo "End: $(date)"