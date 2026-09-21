#!/bin/bash
BASELINE_MODELS=(dlinknet unet)

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

for model in "${BASELINE_MODELS[@]}"; do
    echo "--- baseline model=${model} ---"
    (
      export EXP_TAG="baseline_${model}"
      export FIT_CONFIG="config/baseline/${model}.yml"
      export HPO_CONFIG="config/baseline/hpo_${model}.yml"
      source "$REPO_DIR/scripts/hpc/tm/_stages.sh"
    )
done
echo "### all baseline DONE ###"