#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --job-name="tm-combo"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-user=lxxcat004@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --output=/home/lxxcat004/InstaRoad/InstaRoadPrototype/logs/slurm-%j.out

#
# TerraTorch dispatcher: same --SCRIPT= + KEY=VALUE interface as the unet
# train_both.sbatch, pointed at scripts/hpc/tm/*.sh experiments. STAGE is a
# plain KEY=VALUE here (tune | fit | both) — tm/_stages.sh handles all three
# itself, so there's no separate "_both" sbatch to maintain.
#
#   sbatch scripts/hpc/tm/train_tm.sbatch --SCRIPT=tm/dscnet.sh STAGE=tune
#   sbatch scripts/hpc/tm/train_tm.sbatch --SCRIPT=tm/dscnet.sh STAGE=fit
#   sbatch scripts/hpc/tm/train_tm.sbatch --SCRIPT=tm/dscnet.sh STAGE=both SEED=1 MAX_EPOCHS=100
#   sbatch scripts/hpc/tm/train_tm.sbatch --SCRIPT=tm/dscnet.sh STAGE=fit FIT_CONFIG=other.yml
#
# --gres=gpu:1 by default; override on the command line (e.g.
# `sbatch --gres=gpu:2 scripts/hpc/tm/train_tm.sbatch ...`) if your HPO_CONFIG
# hands terratorch-iterate a Ray cluster spanning more GPUs.
SCRIPT=""

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

for arg in "$@"; do
  case "$arg" in
    --SCRIPT=*|--script=*) SCRIPT="${arg#*=}" ;;
    *=*)                   export "$arg" ;;   # KEY=VALUE -> env override (STAGE, SEED, FIT_CONFIG, HPO_CONFIG, MAX_EPOCHS, STRATEGY, ...)
    *)                      echo "WARN: ignoring unrecognized arg '$arg'." >&2 ;;
  esac
done

if [ -z "$SCRIPT" ]; then
  echo "usage: sbatch scripts/hpc/tm/train_tm.sbatch --SCRIPT=<script.sh> [STAGE=tune|fit|benchmark|both] [KEY=VALUE ...]" >&2
  exit 2
fi

# Resolve the inner script: names (e.g. dscnet.sh) resolve under
# scripts/hpc/tm/; an existing literal/absolute path is used as given.
if [ -f "$REPO_DIR/scripts/hpc/$SCRIPT" ]; then
  SCRIPT_PATH="$REPO_DIR/scripts/hpc/$SCRIPT"
elif [ -f "$SCRIPT" ]; then
  SCRIPT_PATH="$SCRIPT"
else
  echo "ERROR: inner script not found: $SCRIPT (looked in $REPO_DIR/scripts/hpc/)" >&2
  exit 2
fi

echo "### train_tm.sbatch: $SCRIPT_PATH  STAGE=${STAGE:-fit} ###"
# --- multi-seed support: SEED=0,1,2,42 -> one fit/benchmark per seed, ---
# --- all sharing one ARTEFACTS_DIR via a pinned STAMP ------------------
if [[ "${SEED:-}" == *,* ]]; then
  IFS=',' read -ra SEED_LIST <<< "$SEED"

  echo "### train_tm.sbatch: multi-seed run  $SCRIPT_PATH  STAGE=${STAGE:-fit}  seeds=${SEED_LIST[*]} ###"
  for s in "${SEED_LIST[@]}"; do
    echo "### train_tm.sbatch: seed=$s ###"
    SEED="$s" bash "$SCRIPT_PATH"
  done
  echo "### train_tm.sbatch: DONE (multi-seed) ###"
  exit 0
fi


bash "$SCRIPT_PATH"
echo "### train_tm.sbatch: DONE ###"

