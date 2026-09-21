#!/bin/bash
EXP_TAG="${EXP_TAG:-unet}"

FIT_CONFIG="${FIT_CONFIG:-config/decoder/unet_decoder.yml}"
HPO_CONFIG="${HPO_CONFIG:-config/hpo_decoder/hpo_unet.yml}"

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

source "$REPO_DIR/scripts/hpc/tm/_stages.sh"