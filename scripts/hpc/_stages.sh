#!/bin/bash
# Shared tune/fit/benchmark engine for the TerraTorch (finetuning_tm)
# experiment scripts. NOT submitted directly — each experiment script sets
# its config (EXP_TAG, FIT_CONFIG, HPO_CONFIG) and sources this file.
#
# Save as: scripts/hpc/tm/_stages.sh
#MAX_EPOCHS= num 
# STAGE=tune  terratorch-iterate HPO search (Optuna, optionally Ray) driven
#             by HPO_CONFIG. MLflow (sqlite, under this run's RUN_DIR) tracks
#             every trial.
# STAGE=fit   terratorch fit on FIT_CONFIG, then benchmark (terratorch test)
#             the checkpoint it produced on the same config's test split.
# STAGE=benchmark  terratorch test on FIT_CONFIG's test split
#
# Replication contract: everything an experiment needs lives in its script +
# this engine; the knobs meant to vary at submit time are SEED, STAGE, and
# the two config paths (FIT_CONFIG / HPO_CONFIG).
#
# NOTE on bridging tune -> fit: terratorch-iterate's HPO search does not
# automatically hand you a "best_params" file the way the unet Optuna loop
# does. To fit the winning trial you have two options (both manual — this
# engine can't pick one for you without knowing your HPO_CONFIG's layout):
#   (a) copy the winning trial's hyperparameters into FIT_CONFIG, then run
#       STAGE=fit, or
#   (b) run `terratorch iterate --repeat --config <hpo config>
#       --parent_run_id <mlflow run id from the search>` yourself

# STAGE=both therefore runs tune, then fit against whatever FIT_CONFIG
# already points to — update FIT_CONFIG between searches as needed.
set -euo pipefail

# --- relevant directories --------
stamp="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
USER_NAME="${USER:-$(whoami)}"
REPO_DIR="${REPO_DIR:-$HOME/InstaRoad/InstaRoadPrototype}"
VENV_DIR="${VENV_DIR:-$HOME/InstaRoad/InstaRoadPrototype/.venv}"
FINETUNE_DIR="${FINETUNE_DIR:-$REPO_DIR/src/finetuning_tm}"

: "${EXP_TAG:?experiment script must set EXP_TAG (e.g. dscnet)}"

# controlling artefact
ARTEFACTS_DIR="${ARTEFACTS_DIR:-$FINETUNE_DIR/artefacts/tm_${EXP_TAG}_${stamp}}"
WANDB_DIR_ROOT="${ARTEFACTS_DIR}/wandb"
CONFIGS_DIR="${ARTEFACTS_DIR}/configs"
HPO_OUTPUT_DIR="${ARTEFACTS_DIR}/hpo_output"  # terratorch-iterate's --repeat results 
mkdir -p "$WANDB_DIR_ROOT" "$CONFIGS_DIR" "$HPO_OUTPUT_DIR"

# --- configs (set by experiment script; overridable at submit time) --------
STAGE="${STAGE:-fit}"                 
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PRECISION="${PRECISION:-bf16-mixed}"  
DATA_CONFIG="${DATA_CONFIG:-$REPO_DIR/src/finetuning_tm/config/rosa_data/base_data.yml}"  
CKPT=${CKPT:-}  
FIT_CONFIG="${FIT_CONFIG:-}"        
HPO_CONFIG="${HPO_CONFIG:-}"  

# -----------------------------------------------------------------------------


RUN_DIR="${ARTEFACTS_DIR}/runs/seed${SEED}_stage${STAGE}_${stamp}"
printf '\n\n'
echo "run dir is $RUN_DIR"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/${STAGE}_${stamp}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"
echo "host=$(hostname)  exp=tm/${EXP_TAG}  stage=${STAGE}  seed=${SEED}"

if [ ! -d "$FINETUNE_DIR" ]; then
  echo "ERROR: ${FINETUNE_DIR} not visible on $(hostname). Is /scratch mounted / repo checked out?" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="${FINETUNE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# wandb creates its "wandb/" run dir inside whatever WANDB_DIR points to
export WANDB_DIR="$ARTEFACTS_DIR"

echo "python=$(which python)"
echo "terratorch=$(which terratorch)"
printf '\n\n'
#============================== helper methods ===================================
# Resolve a config path: as given, or relative to FINETUNE_DIR.
resolve_config () {   # $1=var name (FIT_CONFIG|HPO_CONFIG)  $2=human label for errors
  local val="${!1}"
  if [ -z "$val" ]; then
    echo "ERROR: ${1} not set — experiment script must set it to run STAGE=${2}." >&2
    exit 1
  fi
  if [ -f "$val" ]; then readlink -f "$val"; return; fi
  if [ -f "$FINETUNE_DIR/$val" ]; then readlink -f "$FINETUNE_DIR/$val"; return; fi
  echo "ERROR: ${2} config '${val}' not found (checked as-is and under ${FINETUNE_DIR})." >&2
  exit 1
}

# Resolve a checkpoint: $CKPT if given, else the newest under ARTEFACTS_DIR.
# Prefers a monitored checkpoint over lightning's rolling `last.ckpt`.
resolve_ckpt () {
  if [ -n "${CKPT:-}" ]; then
    if [ ! -f "$CKPT" ]; then
      echo "ERROR: CKPT='${CKPT}' not found." >&2
      exit 1
    fi
    readlink -f "$CKPT"
    return
  fi

  local found
  found="$(find "$ARTEFACTS_DIR" -name '*.ckpt' ! -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
           | sort -rn | head -n1 | cut -d' ' -f2-)" || true
  if [ -z "$found" ]; then
    found="$(find "$ARTEFACTS_DIR" -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
             | sort -rn | head -n1 | cut -d' ' -f2-)" || true
  fi
  if [ -z "$found" ]; then
    echo "ERROR: no .ckpt under ${ARTEFACTS_DIR} and CKPT not set." >&2
    exit 1
  fi
  readlink -f "$found"
}

# ============================== STAGE: tune ==================================
run_tune () {
  local cfg; cfg="$(resolve_config HPO_CONFIG tune)"

  local mlflow_uri="sqlite:///${RUN_DIR}/mlflow.db"   # RUN_DIR is absolute -> 4 slashes
  local run_cfg="${RUN_DIR}/$(basename "$cfg")"

  # single source of truth: the config iterate reads and the env var lightning's
  # MLFlowLogger reads must be the same string, or nested runs are created in one
  # store and looked up in the other.
  python - "$cfg" "$run_cfg" "$mlflow_uri" <<'PY'
import sys, yaml
src, dst, uri = sys.argv[1:4]
d = yaml.safe_load(open(src))
d["storage_uri"] = uri
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
PY

  export MLFLOW_TRACKING_URI="$mlflow_uri"
  export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
  
  echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
  echo "run config     =${run_cfg}"

  cd "$ARTEFACTS_DIR"
  terratorch iterate --hpo \
    --config "$run_cfg" \
    --output_path "$HPO_OUTPUT_DIR" \
    --custom_modules_path "$FINETUNE_DIR/custom_modules"

  echo "=== HPO DONE === "
}

# ============================== STAGE: fit ===================================
run_fit () {
  local cfg; cfg="$(resolve_config FIT_CONFIG fit)"

  echo "=== FIT (terratorch fit) max_epochs=${MAX_EPOCHS} precision=${PRECISION} config=${cfg} ==="
  cd "$ARTEFACTS_DIR"   # relative paths in FIT_CONFIG (checkpoints/, loggers, etc.) land here now


  terratorch fit \
    --config "$cfg" \
    --custom_modules_path "$FINETUNE_DIR/custom_modules" \
    --trainer.max_epochs "$MAX_EPOCHS" \
    --trainer.default_root_dir "$RUN_DIR" \
    --trainer.precision "$PRECISION" \
    --data "$DATA_CONFIG" \
    --seed_everything "$SEED"

  echo "=== FIT DONE ===  logs in ${RUN_DIR}"
}

# ============================== STAGE: benchmark ===================================
run_benchmarking(){
  echo "=== locating checkpoint for benchmarking ==="
  # trainer.default_root_dir steers lightning's default checkpoint dir into
  # RUN_DIR, but a config with its own explicit (relative) ModelCheckpoint
  # dirpath can still write elsewhere under cwd (now ARTEFACTS_DIR) — so
  # search the whole artefacts tree, newest wins.
  local cfg;  cfg="$(resolve_config FIT_CONFIG benchmark)"
  local ckpt; ckpt="$(resolve_ckpt)"

  echo "using checkpoint: ${ckpt}"

  echo "=== BENCHMARK (terratorch test) ckpt=$(basename "$ckpt") ==="
  terratorch test \
    --config "$cfg" \
    --trainer.precision="${PRECISION}" \
    --custom_modules_path "$FINETUNE_DIR/custom_modules" \
    --ckpt_path "$ckpt"

local dest_cfg_dir="${CONFIGS_DIR}/${EXP_TAG}_seed${SEED}_$stamp"
  mkdir -p "$dest_cfg_dir"
  cp "$cfg" "$dest_cfg_dir/" 2>/dev/null || true
  echo "config copied to ${dest_cfg_dir}"
  echo "=== BENCHMARK DONE ===  logs in ${RUN_DIR}"
}

# ============================== STAGE: prediction ===================================
run_prediction () {
  local cfg;   cfg="$(resolve_config FIT_CONFIG predict)"
  local ckpt;  ckpt="$(resolve_ckpt)"
  local panel_dir="${ARTEFACTS_DIR}/predictions/panels_${stamp}"
  local raster_dir="${ARTEFACTS_DIR}/predictions/rasters_${stamp}"
  mkdir -p "$panel_dir" "$raster_dir"

  echo ""
  echo "=== PREDICT (terratorch predict) config=${cfg} ==="
  echo "    ckpt    = ${ckpt}"
  echo "    panels  = ${panel_dir}"
  echo "    rasters = ${raster_dir}"

  cd "$ARTEFACTS_DIR"   # keeps cwd off any dir containing custom_modules/

  # One atomic JSON value replaces the config's callback list outright —
  # no ordering subtleties with trailing --trainer.callbacks.init_args.* flags.
  local writer_json
  writer_json=$(cat <<JSON
[{"class_path": "custom_modules.callbacks.PredictionWriter",
  "init_args": {"output_dir": "${panel_dir}",
                "rgb_key": "S2L2A",
                "rgb_indices": [0, 1, 2],
                "positive_class": 1,
                "filename_stat": "iou",
                "threshold": 0.5}}]
JSON
)

  terratorch predict \
    --config "$cfg" \
    --custom_modules_path "$FINETUNE_DIR/custom_modules" \
    --ckpt_path "$ckpt" \
    --predict_output_dir "$raster_dir" \
    --trainer.logger=false \
    --trainer.accelerator="${ACCELERATOR:-auto}" \
    --trainer.devices="${DEVICES:-auto}" \
    --trainer.callbacks="$writer_json"

  cp "$cfg" "${RUN_DIR}/" 2>/dev/null || true
  echo "=== PREDICT DONE ===  panels in ${panel_dir}"
}

# ============================== cases ===================================
case "$STAGE" in
  tune) run_tune ;;
  fit)  run_fit ;;
  benchmark) run_benchmarking ;;
  predict) run_prediction ;;
  *) echo "ERROR: STAGE must be tune, fit, benchmark, predict, got '${STAGE}'." >&2; exit 2 ;;
esac