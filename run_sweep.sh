#!/usr/bin/env bash
set -Eeuo pipefail

# Sweep runner for cot-hidden-dynamic.py
# - Iterates over models, pooling methods, accumulation modes, context_aware_k, similarity orders
# - Saves figures and data under OUT_ROOT/<model_name>/<config>/
#
# Customize MODELS below to point to your local HF model folders.

########################
# User configuration
########################

# Data file (edit if needed)
DATA_FILE=${DATA_FILE:-"data/generated_logic_topics.json"}

# Output root directory
OUT_ROOT=${OUT_ROOT:-"results/mid_dataset/sweep_out"}

# Comma-separated sections to include (or 'all')
SECTIONS=${SECTIONS:-"all"}

# Models to evaluate (edit paths as needed)
MODELS=(
  "/home/users/your_name/store/pretrain/Qwen/Qwen3-0.6B"
  "/home/users/your_name/store/pretrain/Qwen/Qwen3-1.7B"
  "/home/users/your_name/store/pretrain/Qwen/Qwen3-4B"
  "/home/users/your_name/store/pretrain/Qwen/Qwen3-8B"

)

# Pooling strategies
POOLINGS=(step_mean)

# Accumulation strategies
ACCUMS=(cumulative)

K_VALUES=(8 16 32)

# Similarity type selector: 0=positions, 1=Δ, 2=Δ², 3=Menger curvature
ORDERS=(0 1 2 3)

# Device detection (override by setting DEVICE env var)
if [[ -z "${DEVICE:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    DEVICE=$(python - <<'PY'
import torch
print('cuda:0' if torch.cuda.is_available() else 'cpu')
PY
)
  else
    DEVICE="cpu"
  fi
fi

########################
# Helpers
########################

safe_name() {
  # sanitize string to filesystem-safe name
  local s="$1"
  s="${s//\//_}"
  s="${s//\\/_}"
  s="${s//:/_}"
  s="${s// /_}"
  s="${s//,/}"
  s="${s//|/_}"
  s="${s//\*/_}"
  s="${s//\?/_}"
  s="${s//$'\n'/_}"
  s="${s//$'\t'/_}"
  s="${s//(/_}"
  s="${s//)/_}"
  s="${s//[/_}"
  s="${s//]/_}"
  # collapse double underscores
  while [[ "$s" == *"__"* ]]; do s="${s//__/_}"; done
  # trim leading/trailing dots/underscores
  s="${s##[._]}"; s="${s%%[._]}"
  printf "%s" "$s"
}

run_one() {
  local model="$1" pooling="$2" accum="$3" k="$4" order="$5"
  local model_name
  model_name=$(basename "$model")

  # Only sweep K for context_aware_mean; for others, keep K=0 to reduce combos
  if [[ "$pooling" != "context_aware_mean" ]]; then
    k=0
  fi

  local cfg_dir="pool=${pooling}_acc=${accum}_k=${k}_ord=${order}"
  local save_dir="${OUT_ROOT}/$(safe_name "$model_name")/${cfg_dir}"

  # Skip if this configuration directory already exists
  if [[ -d "$save_dir" ]]; then
    echo "[SKIP] Already exists: $save_dir"
    return 0
  fi
  mkdir -p "$save_dir"

  echo "=== Running model=$model_name pooling=$pooling accum=$accum k=$k order=$order device=$DEVICE ==="
  echo "Save dir: $save_dir"

  # Record args for reproducibility
  cat >"$save_dir/args.json" <<JSON
{
  "hf_model": "${model}",
  "data_file": "${DATA_FILE}",
  "pooling": "${pooling}",
  "accumulation": "${accum}",
  "context_aware_k": ${k},
  "similarity_order": ${order},
  "sections": "${SECTIONS}",
  "device": "${DEVICE}",
  "save_dir": "${save_dir}"
}
JSON

  # Run and log
  python cot-hidden-dynamic.py \
    --hf_model "$model" \
    --data_file "$DATA_FILE" \
    --pooling "$pooling" \
    --accumulation "$accum" \
    --context_aware_k "$k" \
    --similarity_order "$order" \
    --sections "$SECTIONS" \
    --save_dir "$save_dir" \
    --device "$DEVICE" \
    >"$save_dir/run.log" 2>&1 || {
      echo "Run failed for model=$model_name cfg=$cfg_dir. See $save_dir/run.log" >&2
      return 1
    }
}

########################
# Main sweep
########################

mkdir -p "$OUT_ROOT"

for model in "${MODELS[@]}"; do
  if [[ ! -d "$model" ]]; then
    echo "[WARN] Model path not found: $model — skipping" >&2
    continue
  fi
  for pooling in "${POOLINGS[@]}"; do
    for accum in "${ACCUMS[@]}"; do
      if [[ "$pooling" == "context_aware_mean" ]]; then
        for k in "${K_VALUES[@]}"; do
          for order in "${ORDERS[@]}"; do
            run_one "$model" "$pooling" "$accum" "$k" "$order"
          done
        done
      else
        for order in "${ORDERS[@]}"; do
          run_one "$model" "$pooling" "$accum" 0 "$order"
        done
      fi
    done
  done
done

echo "Sweep complete. Output saved under: $(realpath "$OUT_ROOT")"
