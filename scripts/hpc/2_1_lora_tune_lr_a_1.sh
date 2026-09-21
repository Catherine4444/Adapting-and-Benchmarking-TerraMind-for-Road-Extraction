#!/bin/bash
BASELINE_MODELS=(all mlp attn)

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

for model in "${BASELINE_MODELS[@]}"; do
    echo "--- terramind lora model=${model} ---"
    (
      export EXP_TAG="${model}_a_1"
      export FIT_CONFIG="config/peft/lora_linear_${model}_modules.yml"
      export HPO_CONFIG="config/hpo_peft/1_lora_linear_${model}_modules.yml"
      source "$REPO_DIR/scripts/hpc/tm/_stages.sh"
    )
done
echo "### all terramind lora models DONE ###"