#!/bin/bash
#SBATCH -p Abaki
#SBATCH --qos=abaki
#SBATCH --job-name=repa_render
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/analysis/qualitative/repa_quali_%j.out
#SBATCH --error=slurm/logs/analysis/qualitative/repa_quali_%j.err


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

python -m scripts.qualitative_analysis.qualitative_render