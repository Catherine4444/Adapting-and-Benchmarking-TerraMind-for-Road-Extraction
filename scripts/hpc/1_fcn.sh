#!/bin/bash
EXP_TAG="${EXP_TAG:-fcn}"
FIT_CONFIG="${FIT_CONFIG:-config/decoder/fcn_decoder.yml}"
HPO_CONFIG="${HPO_CONFIG:-config/hpo_decoder/hpo_fcn.yml}"

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

source "$REPO_DIR/scripts/hpc/tm/_stages.sh"