#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --job-name="tm-peft"
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
export DEFAULT_RANKS="${RANKS:-4,16,64}"
SEEDS="${SEEDS:-0,1,2}"
SCRIPTS=(tm/2_2_lora_attn_mod.sh tm/2_2_lora_mlp_mod.sh)

# For fine grained seed setting, override default above here 
declare -A SEED_RANKS
# SEED_RANKS[1]="64"


IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for script in "${SCRIPTS[@]}"; do
  for s in "${SEED_LIST[@]}"; do

    if [[ -n "${SEED_RANKS[$s]:-}" ]]; then # the override
      export RANKS="${SEED_RANKS[$s]}"
    else
      export RANKS="$DEFAULT_RANKS"
    fi

    echo "### ${script} seed=${s} STAGE=${STAGE} ###"
    SEED="$s" bash "$REPO_DIR/scripts/hpc/$script"
  done
done
echo "### all decoders DONE ###"