#!/usr/bin/env bash
set -euo pipefail

board_root="${NEUTTS_BOARD_ROOT:-/home/orangepi/neutts_2e}"
bin_dir="${NEUTTS_BIN_DIR:-$board_root/bin}"
model_name="${NEUTTS_MODEL_NAME:-neutts-2e-Q4_K_M.gguf}"
server_port="${NEUTTS_SERVER_PORT:-8080}"
threads="${NEUTTS_THREADS:-4}"
batch_threads="${NEUTTS_BATCH_THREADS:-4}"
cpu_range="${NEUTTS_CPU_RANGE:-4-7}"
batch_cpu_range="${NEUTTS_BATCH_CPU_RANGE:-$cpu_range}"
taskset_range="${NEUTTS_TASKSET_RANGE:-$cpu_range}"
poll="${NEUTTS_POLL:-50}"
poll_batch="${NEUTTS_POLL_BATCH:-1}"
priority="${NEUTTS_PRIORITY:-0}"
repack_args=()

if [[ "${NEUTTS_REPACK:-1}" != "1" ]]; then
  repack_args+=(--no-repack)
fi

export LD_LIBRARY_PATH="$bin_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "${NEUTTS_SPEECH_ONLY_HEAD:-1}" == "1" ]]; then
  export LLAMA_NEUTTS_SPEECH_ONLY=1
  export LLAMA_NEUTTS_SPEECH_CANDIDATES="${NEUTTS_SPEECH_CANDIDATES:-1}"
  export LLAMA_NEUTTS_COMPACT_LOGITS="${NEUTTS_COMPACT_LOGITS:-0}"
else
  unset LLAMA_NEUTTS_SPEECH_ONLY || true
  unset LLAMA_NEUTTS_SPEECH_CANDIDATES || true
  unset LLAMA_NEUTTS_COMPACT_LOGITS || true
fi

exec taskset -c "$taskset_range" "$bin_dir/llama-server" \
  --model "$board_root/models/$model_name" \
  --ctx-size 2048 \
  --threads "$threads" \
  --threads-batch "$batch_threads" \
  --cpu-range "$cpu_range" \
  --cpu-strict 1 \
  --cpu-range-batch "$batch_cpu_range" \
  --cpu-strict-batch 1 \
  --poll "$poll" \
  --poll-batch "$poll_batch" \
  --prio "$priority" \
  --prio-batch "$priority" \
  --parallel 1 \
  "${repack_args[@]}" \
  --device none \
  --host 127.0.0.1 \
  --port "$server_port" \
  --no-webui \
  --warmup
