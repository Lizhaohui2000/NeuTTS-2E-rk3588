#!/usr/bin/env bash
set -euo pipefail

board_root="${NEUTTS_BOARD_ROOT:-/home/orangepi/neutts_2e}"
bin_dir="${NEUTTS_BIN_DIR:-$board_root/bin}"
python_bin="${NEUTTS_PYTHON:-python3}"
cpu_range="${NEUTTS_TASKSET_RANGE:-4-7}"
export LD_LIBRARY_PATH="$bin_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${NEUTTS_PYTHONPATH:-$board_root/site-packages}${PYTHONPATH:+:$PYTHONPATH}"

exec taskset -c "$cpu_range" "$python_bin" \
  "$board_root/scripts/neucodec_rk3588_split_runtime.py" \
  --llama-cli "$bin_dir/llama-cli" \
  --model "$board_root/models/neutts-2e-Q4_K_M.gguf" \
  --speakers "$board_root/scripts/speakers.json" \
  --model-dir "$board_root/models_dynamic" \
  --dynamic \
  --npu-core-mask "${NEUTTS_NPU_CORE_MASK:-core012}" \
  --speech-only-head \
  "$@"
