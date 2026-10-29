# Multi-server pipeline parallelism

## Scope and rank contract

The MP connector maps one vLLM scheduler's workers onto equally sized,
contiguous rank ranges, ordered by `lmcache.mp.server_urls`. This document
covers `n_servers > 1` and `PP > 1`. The legacy single-server and TP-only
paths keep their existing key conventions.

Supported geometry requires DP=1, PCP=1, `world_size = TP * PP`,
`world_size % n_servers = 0`, `TP % DCP = 0` and `world_size <= 255`.
PCP is rejected independently of DCP: it adds an unmodeled rank axis even
when DCP=1. The connector rejects it before creating network adapters;
`ParallelStrategy` also rejects inconsistent world sizes or invalid ranks
for direct adapter callers.

Each route is a nonempty intersection of a server's rank range and a PP
stage. Replicated MLA splits this intersection further by DCP shard.
For example, TP=4, PP=3, two servers:

```text
stage 0: rank 0 1 2 3   stage 1: rank 4 5 | 6 7   stage 2: rank 8 9 10 11
         server 0 -----------------------|-------------------- server 1

scheduler -- lookup(stage0, ranks0..3) --> server 0
          -- lookup(stage1, ranks4,5) --> server 0
          -- lookup(stage1, ranks6,7) --> server 1
          -- lookup(stage2, ranks8..11) -> server 1
```

One server may hold several stages; one stage may span servers. Endpoint
ordering describes deployment placement, not an automatic discovery of GPU
hosts. For GPU transfer, each server needs access to its workers' local IPC
resources; shared L2 provides cache sharing across nodes.

## Cache identity and layout ownership

The connector decorates its base model identity with a cache generation tag
covering revision, dtype, cache dtype, DCP, the MLA sharing option and the
deployment-controlled `lmcache.mp.cache_namespace`. It appends a stage suffix
encoding TP, PP and stage number. Workers register under the same stage
identity the scheduler uses for that route's lookup.

`LayoutDescRegistry` therefore does not overwrite one stage's layout with
another stage's layout. Repeated registrations within a stage must agree on
layout, object-group descriptors and windows. A rejected registration must
leave the previous descriptor and registration count intact.

The generation tag is not a fingerprint of weights, physical formats or
actual layer boundaries. Change `cache_namespace` when changing those while
retaining the same model identity. KV repartitioning between different TP/PP
configurations is not implemented.

## KV pieces and reader counts

By default each worker owns a distinct piece: `kv_world_size = world_size`,
`kv_worker_id = vllm_rank`, one writer and one reader per piece. The scheduler
sends `selected_worker_ids` for its route; a server queries only those pieces.
An empty selection retains the legacy all-ranks meaning.

For verified pure MLA with `lmcache.mp.replicated_mla=true`, one object is
shared within each local route and DCP shard. Its piece ID is
`pp_rank * DCP + dcp_rank`. The lowest rank of that route writes it; the
lookup reserves exactly `len(route.ranks)` read locks. Every reader releases
one share after its retrieve. A complete hit requires all routes.

DeepSeek V4 always stores per worker. Architectures can mix replicated MLA
with rank-specific or compressed state, so `use_mla` alone does not establish
that the complete cache object is replicated.

## Sessions and scheduler telemetry

Session identity is `mp-pp-v1:<length>:<request_id>:p<stage>:d<shard>`, where
`all` represents worker-owned KV rather than replicated MLA. The length
prefix makes an original request ID containing delimiters unambiguous.
Lookup, prefetch status, store/retrieve, free-lock and end-session operations
use this same qualified identity.

L0 block allocation telemetry describes the scheduler's block pool, so it is
reported once per physical endpoint, not once per stage/DCP route. The last
route for each endpoint in stable construction order is its representative
for endpoint-level operations. Allocation records are copied to the canonical
`BlockAllocationRecord`, with that representative route's qualified request
ID. The model label remains the scheduler-wide model label.

The normal END_SESSION fan-out includes this representative ID. Consequently
`MP_VLLM_END_SESSION` removes the same owner that allocation telemetry added,
without a new RPC field or decoding a reserved prefix from arbitrary user IDs.
The scheduler's original records are unchanged. Other route completion events
have no matching allocation owner and are harmless to that subscriber.

This preserves existing flat L0 telemetry's limitations: it is not a separate
per-GPU/per-stage block allocator measurement. CLEAR and allocation reporting
deduplicate by physical endpoint; lookup, free-lock and session completion
fan out by route.

## Aggregation, cleanup and recovery

1. Submit one LOOKUP per route. Retain acknowledgement futures before polling
   status so an early status request cannot consume a nonexistent job.
2. Aggregate complete route results by their minimum prefix length. Release
   the over-hit tail only on longer routes, preserving each route's key,
   selected pieces, session ID and reader count.
3. Free the APC-overlapping prefix using route-specific keys. Worker retrieve
   releases the reader's remaining shares after copying.
4. A request cancelled before lookup completion drains its outstanding
   acknowledgements/status RPCs before sending route END_SESSION messages.
   Timeouts retain the existing degraded-mode and TTL behavior.
5. Any unhealthy route makes the scheduler unhealthy. Workers re-register
   local KV contexts after their server's heartbeat recovery. Restart does
   not restore L1; refill or shared L2 is needed for later cache hits.

This is not distributed transaction or high-availability failover support.
Long queue waits beyond read-lock TTL, cancellation during an asynchronous
retrieve and node loss require further fault-oriented model tests. The
stream-disconnect integration case cancels after generated text arrives; it
does not deterministically inject an abort inside a GPU retrieve.

## Protocol capability and unsupported layouts

Every server must advertise `mp_partitioned_pp_v1`; only the merged-object
configuration advertises it. `selected_worker_ids` is an additive dataclass
field and protobuf field 11. Builds regenerate gRPC bindings; updating a
`.pyi` alone cannot update the runtime wire schema.

Independent sliding-window/recurrent object groups can have non-contiguous
sets of recoverable boundaries. Taking the minimum of local maxima is then
unsound. Supporting them needs a cross-route candidate intersection and
prepare/lock step, so `--separate-object-groups` is rejected. Multi-group or
DSV4 engine-driven transfer is also rejected; the existing LMCache-driven
GPU path handles merged groups. Mamba/GDN models that require separate
objects are outside this supported configuration.

See [hybrid KV groups](hybrid-kv-cache-groups.md),
[worker liveness](../../v1/multiprocess/worker_liveness.md) and
[request transport](../../v1/multiprocess/transport/request_transport.md).

## Regression coverage

- `test_mp_connector_autostart.py`: both connector roles reject PCP (with
  DCP=1 or DCP>1) and inconsistent world size before network adapters.
- `test_vllm_mp_parallel_topology.py`: route coverage, MLA writer/reader
  balance, stage/session identities, and invalid direct strategies.
- `test_vllm_mp_adapter.py`: real constructor with transport doubles;
  endpoint CLEAR deduplication, telemetry owner cleanup through the actual
  L0 subscriber, asymmetric hits, and cancel-before-ack cleanup.
- `test_grpc_transport.py`: real gRPC round trip explicitly checks selection
  and reader fields, which are excluded from dataclass equality.
- `multi-server-pp.sh`: a two-GPU model workload covering cold/hit equality,
  optional L2 reload, asymmetric L1 misses and tail-lock release, stream
  disconnect/session cleanup, and a single-server restart followed by refill
  and a fresh cache hit. L2-configured runs skip the L1-only asymmetry case.

Run commands and deployment caveats are in the user-facing
`docs/source/mp/multi_server_pp.rst`. These are executable regression cases;
adding a case does not establish that it passed in a particular deployment.
