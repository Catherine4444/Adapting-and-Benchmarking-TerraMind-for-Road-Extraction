#!/bin/bash
# 
EXP_TAG="linear_precision_32-true"
PRECISION="${PRECISION:-32-true}"  

FIT_CONFIG="${FIT_CONFIG:-config/decoder/linear_decoder.yml}"
HPO_CONFIG="${HPO_CONFIG:-config/hpo_decoder/hpo_linear.yml}"

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

source "$REPO_DIR/scripts/hpc/tm/_stages.sh"