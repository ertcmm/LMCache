#!/usr/bin/env bash
# End-to-end KV correctness for one LMCache MP server per vLLM PP stage.
# The two servers are independent processes on the same test pod; the model
# and request API are intentionally ordinary dense vLLM so the test isolates
# LMCache's PP routing rather than a model-specific KV layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_PP_PORT="${LMCACHE_PP_PORT:-6556}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
L1_SIZE_GB="${L1_SIZE_GB:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1200}"
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-10}"
MAX_TOKENS="${MAX_TOKENS:-16}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
PP_DIR="${RESULTS_DIR}/multi_server_pp"
SERVER_A_LOG="/tmp/build_${BUILD_ID}_lmcache_pp0.log"
SERVER_B_LOG="/tmp/build_${BUILD_ID}_lmcache_pp1.log"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm_multi_server_pp.log"
PROMPT_FILE="${PP_DIR}/prompt.txt"
OUTPUT_COLD="${PP_DIR}/output_cold.txt"
OUTPUT_HIT="${PP_DIR}/output_lmcache_hit.txt"
mkdir -p "$PP_DIR"

if [[ "${LMCACHE_REQUEST_TRANSPORT:-zmq}" != "zmq" ]]; then
    echo "multi_server_pp currently requires ZMQ request transport" >&2
    exit 2
fi
if [[ "$LMCACHE_PORT" == "$LMCACHE_PP_PORT" ]]; then
    echo "The two LMCache servers must use distinct request ports" >&2
    exit 2
fi
if [[ "${VERIFY_L2_RETRIEVAL:-0}" =~ ^(1|true|yes)$ ]] \
    && [[ -z "${LMCACHE_L2_ADAPTER_JSON_A:-${LMCACHE_L2_ADAPTER_JSON:-}}" \
        || -z "${LMCACHE_L2_ADAPTER_JSON_B:-${LMCACHE_L2_ADAPTER_JSON:-}}" ]]; then
    echo "VERIFY_L2_RETRIEVAL requires an L2 adapter config for both servers" >&2
    exit 2
fi

echo "=== Multi-server PP LMCache correctness test ==="
echo "Model: ${MODEL} | TP=1 PP=2 | servers: ${LMCACHE_PORT},${LMCACHE_PP_PORT}"
echo "Results: ${PP_DIR}"

launch_lmcache_server() {
    local port="$1" http_port="$2" log_file="$3" l2_json="$4"
    local -a l2_args=()
    if [[ -n "$l2_json" ]]; then
        l2_args=(--l2-adapter "$l2_json")
    fi
    lmcache server \
        --transport zmq \
        --host localhost \
        --port "$port" \
        --http-host localhost \
        --http-port "$http_port" \
        --chunk-size "$CHUNK_SIZE" \
        --l1-size-gb "$L1_SIZE_GB" \
        --eviction-policy LRU \
        --max-workers 4 \
        "${l2_args[@]}" \
        > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" >> "$PID_FILE"
    echo "$pid"
}

wait_for_mp_server() {
    local port="$1"
    SERVER_URL="tcp://localhost:${port}" python3 - <<'PY'
import os
import sys
import time

import zmq

from lmcache.integration.vllm.mp_server_launcher import is_mp_server_healthy

url = os.environ["SERVER_URL"]
context = zmq.Context()
try:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if is_mp_server_healthy(url, context, timeout=2.0):
            print(f"LMCache MP server healthy at {url}")
            break
        time.sleep(2)
    else:
        print(f"LMCache MP server did not become healthy at {url}", file=sys.stderr)
        sys.exit(1)
finally:
    context.term()
PY
}

SERVER_A_PID="$(launch_lmcache_server \
    "$LMCACHE_PORT" "$LMCACHE_HTTP_PORT" "$SERVER_A_LOG" \
    "${LMCACHE_L2_ADAPTER_JSON_A:-${LMCACHE_L2_ADAPTER_JSON:-}}")"
SERVER_B_PID="$(launch_lmcache_server \
    "$LMCACHE_PP_PORT" "$((LMCACHE_HTTP_PORT + 1))" "$SERVER_B_LOG" \
    "${LMCACHE_L2_ADAPTER_JSON_B:-${LMCACHE_L2_ADAPTER_JSON:-}}")"
echo "Started LMCache PP0/PP1 servers (PIDs ${SERVER_A_PID}/${SERVER_B_PID})."
wait_for_mp_server "$LMCACHE_PORT"
wait_for_mp_server "$LMCACHE_PP_PORT"

python3 - "$PROMPT_FILE" <<'PY'
import sys

paragraph = (
    "A reliable distributed cache stores independently addressable pieces of "
    "state and restores them with the same identity used during insertion. "
    "Pipeline parallel inference assigns consecutive groups of model layers "
    "to stages, and every stage must recover its own key and value tensors. "
)
with open(sys.argv[1], "w") as output:
    output.write(paragraph * 90)
    output.write("The final repeated sentence continues with reliable state")
PY

BUILD_ID="$BUILD_ID" \
SERVER_URLS="tcp://localhost:${LMCACHE_PORT},tcp://localhost:${LMCACHE_PP_PORT}" \
    python3 - <<'PY' > "${PP_DIR}/kv-transfer-config.json"
import json
import os

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "lmcache.mp.server_urls": os.environ["SERVER_URLS"],
                "lmcache.mp.mp_transfer_mode": "lmcache_driven",
                "lmcache.mp.mq_timeout": 60,
                # Keep persistent Mooncake data from earlier CI/test runs out
                # of the cold-fill phase of this run.
                "lmcache.mp.cache_namespace": (
                    f"multi-server-pp-{os.environ['BUILD_ID']}"
                ),
            },
        }
    )
)
PY
KV_TRANSFER_CONFIG="$(cat "${PP_DIR}/kv-transfer-config.json")"

saved_vllm_port="$VLLM_PORT"
unset VLLM_PORT
VLLM_SERVER_DEV_MODE=1 \
VLLM_BATCH_INVARIANT=1 \
PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --enable-prefix-caching \
    --no-async-scheduling \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$saved_vllm_port" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" >> "$PID_FILE"
export VLLM_PORT="$saved_vllm_port"

if ! wait_for_server "$VLLM_PORT" "$VLLM_READY_TIMEOUT" "$VLLM_LOG"; then
    echo "vLLM PP=2 failed to start. Recent log:"
    tail -n 120 "$VLLM_LOG" || true
    exit 1
fi

send_completion() {
    local output_file="$1"
    python3 - "$VLLM_PORT" "$MODEL" "$PROMPT_FILE" "$MAX_TOKENS" "$output_file" <<'PY'
import json
import sys
import urllib.request

port, model, prompt_file, max_tokens, output_file = sys.argv[1:]
body = json.dumps(
    {
        "model": model,
        "prompt": open(prompt_file).read(),
        "temperature": 0,
        "seed": 0,
        "max_tokens": int(max_tokens),
    }
).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)
with open(output_file, "w") as output:
    output.write(result["choices"][0]["text"])
print(f"Wrote {len(result['choices'][0]['text'])} generated characters to {output_file}")
PY
}

echo "=== Cold run: populate both pipeline-stage caches ==="
send_completion "$OUTPUT_COLD"
sleep "$STORE_DRAIN_SECONDS"
count_l2_hits() {
    grep -Ec '0 L1, [1-9][0-9]* L2' "$1" || true
}
L2_HITS_BEFORE_A="$(count_l2_hits "$SERVER_A_LOG")"
L2_HITS_BEFORE_B="$(count_l2_hits "$SERVER_B_LOG")"

echo "=== Clear vLLM APC and repeat to force LMCache retrieval ==="
reset_status="$(curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' \
    --max-time 60 -X POST "http://127.0.0.1:${VLLM_PORT}/reset_prefix_cache")"
if [[ "$reset_status" != "200" ]]; then
    echo "Failed to reset the vLLM prefix cache (HTTP ${reset_status})" >&2
    exit 1
fi
if [[ "${VERIFY_L2_RETRIEVAL:-0}" =~ ^(1|true|yes)$ ]]; then
    for http_port in "$LMCACHE_HTTP_PORT" "$((LMCACHE_HTTP_PORT + 1))"; do
        clear_status="$(curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' \
            --max-time 60 -H 'Content-Type: application/json' \
            -X POST -d '{"tier":"l1","force":false}' \
            "http://127.0.0.1:${http_port}/cache/clear")"
        if [[ "$clear_status" != "200" ]]; then
            echo "Failed to clear L1 on HTTP port ${http_port} (HTTP ${clear_status})" >&2
            exit 1
        fi
    done
    echo "Cleared both L1 pools; the next hit must be served by Mooncake L2."
fi
send_completion "$OUTPUT_HIT"

if ! cmp -s "$OUTPUT_COLD" "$OUTPUT_HIT"; then
    echo "Cold and LMCache-served completions differ." >&2
    diff -u "$OUTPUT_COLD" "$OUTPUT_HIT" || true
    exit 1
fi

server_index=0
for log_file in "$SERVER_A_LOG" "$SERVER_B_LOG"; do
    retrieves="$(grep -ci "Retrieved" "$log_file" || true)"
    if [[ "${retrieves:-0}" -eq 0 ]]; then
        echo "LMCache server did not serve a retrieve: ${log_file}" >&2
        tail -n 120 "$log_file" || true
        exit 1
    fi
    if [[ "${VERIFY_L2_RETRIEVAL:-0}" =~ ^(1|true|yes)$ ]] \
        && ! grep -Eq '0 L1, [1-9][0-9]* L2' "$log_file"; then
        echo "The server did not report an L2 hit after L1 was cleared: ${log_file}" >&2
        tail -n 120 "$log_file" || true
        exit 1
    fi
    if [[ "${VERIFY_L2_RETRIEVAL:-0}" =~ ^(1|true|yes)$ ]]; then
        if [[ "$server_index" -eq 0 ]]; then
            l2_hits_before="$L2_HITS_BEFORE_A"
        else
            l2_hits_before="$L2_HITS_BEFORE_B"
        fi
        l2_hits_after="$(count_l2_hits "$log_file")"
        if [[ "$l2_hits_after" -le "$l2_hits_before" ]]; then
            echo "No new Mooncake L2 hit after the APC/L1 reset: ${log_file}" >&2
            tail -n 120 "$log_file" || true
            exit 1
        fi
    fi
    echo "${log_file}: ${retrieves} retrieve operation(s)"
    server_index=$((server_index + 1))
done

for route in "server0:pp0" "server1:pp1"; do
    if ! grep -Fq "$route" "$VLLM_LOG"; then
        echo "vLLM did not log expected PP route ${route}." >&2
        tail -n 160 "$VLLM_LOG" || true
        exit 1
    fi
done

echo "PASS: TP=1 PP=2 stored/retrieved KV through both LMCache servers; outputs match."
echo "Logs: ${VLLM_LOG}, ${SERVER_A_LOG}, ${SERVER_B_LOG}"
