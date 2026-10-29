Multi-server pipeline parallelism
================================

``LMCacheMPConnector`` supports routing a pipeline's stages to multiple MP
servers. Each server owns an equal-sized, contiguous range of vLLM ranks,
in the order of ``lmcache.mp.server_urls``. A stage can span servers, and
one server can own ranks from multiple stages.

Requirements and limits
-----------------------

* DP and prefill context parallelism (PCP) must both be 1. DCP can subdivide
  TP; its block/interleave requirements still apply.
* ``world_size == TP * PP <= 255``, and the server count must divide it.
* All servers need matching chunk sizes and the ``mp_partitioned_pp_v1``
  capability. Start servers independently; multi-server auto-start is not
  supported.
* GPU servers must have access to their assigned workers' GPU IPC resources.
  In a multi-node deployment, place a local server with each set of workers
  and use shared L2 for cross-node cache reuse. Two servers in one Pod do
  not validate cross-node IPC placement or independent container isolation.
* Use merged objects (``--no-separate-object-groups``). Mamba/GDN and other
  models requiring separate object groups are outside this configuration.
  DSV4/multi-group layouts require LMCache-driven GPU transfer; DSV4 KV is
  stored per worker. Model startup and KV correctness still need model-level
  validation on hardware with sufficient memory.
* Change ``lmcache.mp.cache_namespace`` when changing weights, layer partition
  or incompatible KV layouts without changing the model identity. The cache
  tag does not hash all of those properties automatically.

See :doc:`configuration` for connector options and DCP alignment rules.

Unit and transport regression tests
-----------------------------------

Use a source checkout with LMCache and its test dependencies installed, and
a compatible vLLM installation for the connector initialization cases. The
gRPC cases also require regenerated bindings (normally generated during
package installation):

.. code-block:: bash

   python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
   python -m pytest -q \
       tests/v1/test_mp_connector_autostart.py \
       tests/v1/test_vllm_mp_parallel_topology.py \
       tests/v1/test_vllm_mp_adapter.py \
       tests/v1/multiprocess/test_grpc_transport.py

Without vLLM, connector-specific cases skip; that is not evidence of their
passing. These tests cover PCP rejection, rank geometry, telemetry allocation
and completion identity, physical-server CLEAR deduplication, asymmetric
lookup tails, cancellation before lookup acknowledgement and gRPC selection
serialization.

Two-GPU integration workload
----------------------------

Run in an otherwise idle test Pod with two GPUs. The harness starts two
LMCache servers and a TP=1/PP=2 vLLM engine, records their PIDs and cleans
them up on exit. From the repository root, use the lower-level entry point
with the already installed LMCache/vLLM environment:

.. code-block:: bash

   MODEL=/mnt/workspace/models/public-models/llm/DeepSeek-R1-Distill-Qwen-14B \
   LMCACHE_PORT=16555 LMCACHE_PP_PORT=16556 \
   LMCACHE_HTTP_PORT=18080 VLLM_PORT=18000 \
   L1_SIZE_GB=8 GPU_MEMORY_UTILIZATION=0.85 \
   bash .buildkite/k3_tests/multiprocess/scripts/run-single-test.sh multi_server_pp

The CI entry point ``multiprocess/run.sh`` additionally runs environment
setup and dependency installation. The command above skips that setup;
ensure the checkout's scripts retain their executable permissions.

The default workload checks:

1. Nonempty, identical greedy outputs on a cold request and a cache hit,
   with both PP servers reporting a retrieve.
2. With no L2 configured, clear only PP1's L1. Require the unequal-hit
   aggregation path, correct recomputation, and no remaining L1 objects
   after a non-forced clear (detecting retained tail locks).
3. Disconnect a streaming client after generated text arrives, wait for
   sessions, prefetch jobs, read/write locks and L2 stores to settle, and
   verify a subsequent completion.
4. Restart only PP1's server, require worker re-registration, then compare
   outputs and require a subsequent retrieve from the replacement process.

Set ``PP_EXTENDED_CHECKS=0`` for the cold/hit smoke case only.
``RECOVER_TIMEOUT`` controls re-registration polling (default 150 seconds).
Cancellation here exercises a live generation disconnect; it does not
deterministically interrupt a GPU copy or test long queue waits beyond TTL.

To require shared-L2 retrieval, supply your test cluster's adapter JSON for
both servers and enable the L2 assertion:

.. code-block:: bash

   export LMCACHE_L2_ADAPTER_JSON_A='{"type":"mooncake_store", ...}'
   export LMCACHE_L2_ADAPTER_JSON_B='{"type":"mooncake_store", ...}'
   VERIFY_L2_RETRIEVAL=1 \
   MODEL=/path/to/model \
   bash .buildkite/k3_tests/multiprocess/scripts/run-single-test.sh multi_server_pp

Replace the JSON placeholders with complete Mooncake settings and configure
its network environment as required by the deployment. The workload clears
only its test servers' L1 pools and uses a per-run cache namespace; it does
not flush shared L2. The asymmetric L1-only case explicitly skips with L2
configured, because L2 would refill the deliberately emptied stage. Run
both L1-only and L2-enabled variants to cover both paths.

Outputs and generated connector configuration are under
``/tmp/lmcache_ci_results_<build-id>/multi_server_pp/``. Logs are
``/tmp/build_<build-id>_vllm_multi_server_pp.log``,
``/tmp/build_<build-id>_lmcache_pp0.log``,
``/tmp/build_<build-id>_lmcache_pp1.log`` and
``/tmp/build_<build-id>_lmcache_pp1_restarted.log``.
