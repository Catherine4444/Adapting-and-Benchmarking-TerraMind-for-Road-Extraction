#!/bin/bash
EXP_TAG="${EXP_TAG:-lora_linear_attn_modules_rank}"

FIT_CONFIG="${FIT_CONFIG:-config/peft/lora_linear_attn_modules.yml}"

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
FINETUNE_DIR="${FINETUNE_DIR:-$REPO_DIR/src/finetuning_tm}"

source "$REPO_DIR/scripts/hpc/tm/_hpo_lora_rank.sh"