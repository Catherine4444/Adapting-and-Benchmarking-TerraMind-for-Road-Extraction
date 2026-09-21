#!/bin/bash

# --- grid search over LoRA rank -----------------------------------------
# RANKS=4,8,16,32,64 (STAGE=fit|benchmark) runs one full terratorch fit
# (+benchmark) per rank, each patching peft_config_kwargs.r in FIT_CONFIG
# and using its own EXP_TAG so runs land in separate artefacts dirs.
# Bypasses HPO_CONFIG entirely — this replaces the Optuna search with an
# explicit sweep.
if [ -n "${RANKS:-}" ]; then
  case "${STAGE:-fit}" in
    fit|benchmark) : ;;
    *) echo "ERROR: RANKS grid search only supports STAGE=fit|benchmark (got '${STAGE:-fit}')." >&2; exit 2 ;;
  esac

  base_cfg="$FIT_CONFIG"
  if [ -f "$base_cfg" ]; then :;
  elif [ -f "$FINETUNE_DIR/$base_cfg" ]; then base_cfg="$FINETUNE_DIR/$base_cfg";
  else echo "ERROR: FIT_CONFIG '$FIT_CONFIG' not found." >&2; exit 1; fi

  IFS=',' read -ra RANK_LIST <<< "$RANKS"
  grid_dir="${FINETUNE_DIR}/config/tm_${EXP_TAG}_grid_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$grid_dir"

  echo "### ${EXP_TAG}: rank grid search over r=${RANK_LIST[*]} ###"
  for r in "${RANK_LIST[@]}"; do
    rank_cfg="${grid_dir}/r${r}.yml"
    python - "$base_cfg" "$rank_cfg" "$r" <<'PY'
import sys, yaml
src, dst, r = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = yaml.safe_load(open(src))
d["model"]["init_args"]["model_args"]["peft_config"]["peft_config_kwargs"]["r"] = r
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
PY

    echo "--- rank r=${r}: config=${rank_cfg} ---"
    (
      export EXP_TAG="${EXP_TAG}_r${r}"
      export FIT_CONFIG="$rank_cfg"
      source "$REPO_DIR/scripts/hpc/tm/_stages.sh"
    )
  done
  echo "### ${EXP_TAG}: rank grid search DONE ###"

  mkdir -p "${FINETUNE_DIR}/artefacts"
  mv "$grid_dir" "${FINETUNE_DIR}/artefacts/"
  
  exit 0
fi

source "$REPO_DIR/scripts/hpc/tm/_stages.sh"