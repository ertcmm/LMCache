# Multiprocess workloads

Workloads are grouped by their inference-engine compatibility. The
multiprocess harness normally owns the LMCache and inference-engine lifecycle;
a workload owns its requests and assertions.

`common/` contains workloads that use the shared engine-adapter contract:

- `lm-eval.sh`: two `lm_eval` runs with output or score consistency checks.
- `long-doc-qa.sh`: LMCache-versus-baseline long-document performance checks.
- `high-concurrency.sh`: concurrent random-prefill completion validation.
- `deadlock.sh`: high-concurrency deadlock regression.
- `long-doc-qa-l2.sh`: L2 restart, data-flow, and performance validation.
- `restart-recovery.sh`: LMCache server restart and worker re-registration.
- `http-api.sh`: LMCache HTTP API and CLI coverage with a live engine.

`vllm/` contains workloads that currently depend on vLLM-specific commands,
response extensions, endpoints, logs, or model launch flags. A future
`sglang/` directory holds the corresponding SGLang-only coverage. For example,
`mp-autostart-tp2.sh` verifies vLLM-specific connector auto-start behavior and
therefore remains under `vllm/` rather than `common/`.

`vllm/multi-server-pp.sh` starts two LMCache MP servers and a TP=1, PP=2 vLLM
engine. It compares a cold completion with one after clearing vLLM's local
prefix cache and requires both stage-local servers to report a retrieval.
This complements the existing TP=2 autostart and Kimi/DSv4 TP correctness
workloads by exercising the multi-server PP routing path.

The workload also checks asymmetric L1 hits and tail-lock release (with no
L2 configured), a streaming client disconnect followed by session cleanup,
and one-server restart/re-registration followed by a fresh cache hit. These
checks run by default; set `PP_EXTENDED_CHECKS=0` for the original smoke test.
When L2 is configured, the asymmetric L1-only case explicitly skips because
the emptied stage would reload from L2. Set `VERIFY_L2_RETRIEVAL=1` and supply
`LMCACHE_L2_ADAPTER_JSON_A/B` to require L2 hits on both stages. The helper
`vllm/multi_server_pp_checks.py` uses only the Python standard library.

Run through `multiprocess/run.sh multi_server_pp` or
`scripts/run-single-test.sh multi_server_pp` so the harness cleans up all
recorded server and vLLM PIDs, including the replacement server after restart.

The harness converts a test name from underscores to hyphens and discovers the
script automatically. For example, `long_doc_qa` resolves to
`common/long-doc-qa.sh`, while `cache_stats` with `INFERENCE_ENGINE=vllm`
resolves to `vllm/cache-stats.sh`. Only test names that share an implementation
need an explicit alias in `../workload-discovery.sh`.

Common workloads use `ENGINE_PORT`, `ENGINE_BASELINE_PORT`, `MODEL`, and
`RESULTS_DIR`. Engine-specific launch behavior must be exposed by an adapter in
`../engines/`. Each engine's CI matrix should run the common workloads plus the
workloads in its engine-named directory. Tests remain separate Buildkite steps
because their GPU counts, models, timeouts, and retry policies differ.
An adapter may temporarily skip unsupported common workloads through
`ENGINE_COMMON_WORKLOAD_BLACKLIST`; this is an exception list, not a workload
allowlist. Transfer-mode and request-transport support is declared separately
by the adapter, so a common pipeline matrix can safely include engines with
different capabilities.
