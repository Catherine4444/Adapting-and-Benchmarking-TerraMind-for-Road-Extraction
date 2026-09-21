#!/bin/bash
EXP_TAG="${EXP_TAG:-upernet}"

FIT_CONFIG="${FIT_CONFIG:-config/decoder/upernet_decoder.yml}"
HPO_CONFIG="${HPO_CONFIG:-config/hpo_decoder/hpo_upernet.yml}"

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"

source "$REPO_DIR/scripts/hpc/tm/_stages.sh"