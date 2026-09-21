#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --job-name="train-tm-decoders"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-user=lxxcat004@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lxxcat004/InstaRoad/InstaRoadPrototype/logs/slurm-%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

export STAGE="${STAGE:-fit}"
export MAX_EPOCHS="${MAX_EPOCHS:-100}"
SEEDS="${SEEDS:-0,1,2}"
SCRIPTS=(tm/1_upernet.sh tm/1_linear.sh)

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for script in "${SCRIPTS[@]}"; do
  for s in "${SEED_LIST[@]}"; do
    echo "### ${script} seed=${s} STAGE=${STAGE} ###"
    SEED="$s" bash "$REPO_DIR/scripts/hpc/$script"
  done
done
echo "### all decoders DONE ###"