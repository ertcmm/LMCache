# vLLM MP CUDA IPC registration benchmark

This benchmark measures the end-to-end TTFT of a fresh vLLM worker's first
request and summarizes the new timing log records from the vLLM worker and
LMCache MP server. It separates:

- client-side full-registration round trip, including CUDA IPC handle export
  and the server acknowledgement;
- server-side CUDA IPC registration work, including GPU cache context and
  staging setup;
- time the first `STORE` or `RETRIEVE` path blocks waiting for registration;
- first-request TTFT and later-request TTFT on the same vLLM instance.

The benchmark driver uses Python's standard library and an
OpenAI-compatible streaming chat endpoint. Use the same model, MP server,
storage configuration, and GDS settings as the deployment being evaluated.

## Measure the first STORE

1. Start LMCache MP and a fresh vLLM worker. Keep the vLLM log and LMCache
   server log in separate files; use fresh log files for each run. Make sure
   the worker has not received a STORE or RETRIEVE request yet.
2. Supply a representative prompt file. STORE mode replaces each whitespace
   word with a random identifier so later benchmark requests do not hit data
   saved by earlier iterations. Tokenizers can split those identifiers
   differently from the original words, so tune the input file to the desired
   approximate prompt length.
3. Run:

```bash
python benchmarks/mp_registration/benchmark.py \
  --mode store \
  --prompt-file ./representative-prompt.txt \
  --requests 6 \
  --mq-timeout 300 \
  --client-log ./vllm.log \
  --server-log ./lmcache.log \
  --output ./registration-store.json
```

The first request is labeled `first_operation`; the remaining requests are
summarized as steady state. The output includes metadata registration, full
client registration, server CUDA IPC registration, and the first-operation
gate timings from the provided logs. It also reports those full-registration
and gate timings as fractions of `--mq-timeout`.

## Measure the first RETRIEVE

1. With the MP server running, use a source vLLM worker to send the prompt in
   `representative-prompt.txt` and wait for that request to finish storing its
   KV cache.
2. Stop the source worker, then start a fresh vLLM worker with the same model
   and LMCache configuration. Keep the MP server running so the stored cache
   remains available. Disable vLLM's local prefix cache so the target request
   exercises LMCache retrieval.
3. Run the benchmark against the fresh worker with the same prompt file:

```bash
python benchmarks/mp_registration/benchmark.py \
  --mode retrieve \
  --prompt-file ./representative-prompt.txt \
  --requests 6 \
  --mq-timeout 300 \
  --client-log ./fresh-vllm.log \
  --server-log ./lmcache.log \
  --output ./registration-retrieve.json
```

The first request should produce a `RETRIEVE` registration-gate log entry. If
it does not, the run did not exercise first-retrieve registration; check the
worker logs, model identity, prompt, and cache configuration.

## Compare configurations

Run each case with fresh vLLM worker logs and retain the MP server log. Compare
small and large models, then repeat with GDS disabled and enabled. Keep the
vLLM `mq_timeout` consistent with the production value for each case. Compare
the logged full-registration round trip and server duration against that
timeout, and compare `first_request_ttft_delta_vs_steady_p50_ms` to assess the
user-visible first-request cost. Record model/version, GPU, TP size, GDS
settings, prompt file, and `mq_timeout` alongside each JSON result.

The TTFT figures are end-to-end request measurements. The server-side log
duration measures only the `REGISTER_KV_CACHE` handler work; the client-side
round trip also includes local IPC handle preparation and message transport.
