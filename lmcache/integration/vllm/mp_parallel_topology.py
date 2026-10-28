# SPDX-License-Identifier: Apache-2.0
"""Map vLLM pipeline ranks to their local LMCache MP servers."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MPServerRoute:
    """One server-local lookup domain with one exact read-lock count.

    Attributes:
        route_id: Scheduler-local identifier for this route.
        server_url: Request endpoint of the owning MP server.
        server_index: Index of that endpoint in ``server_urls``.
        pp_rank: Pipeline stage whose layers this route stores.
        dcp_rank: Token shard for replicated MLA, or ``None`` when the route
            contains distinct worker-owned KV pieces.
        ranks: vLLM workers that consume the route's objects.
        kv_worker_ids: KV piece ids to query on this server.
        num_kv_readers: Exact number of consumers per queried object.
    """

    route_id: str
    server_url: str
    server_index: int
    pp_rank: int
    dcp_rank: int | None
    ranks: tuple[int, ...]
    kv_worker_ids: tuple[int, ...]
    num_kv_readers: int


def build_mp_server_routes(
    server_urls: Sequence[str],
    tp_size: int,
    pp_size: int,
    dcp_size: int,
    mla_only: bool,
) -> tuple[MPServerRoute, ...]:
    """Build routes for a contiguous, equally divided TP by PP deployment.

    Args:
        server_urls: Endpoints ordered by their contiguous vLLM rank ranges.
        tp_size: Tensor-parallel worker count per pipeline stage.
        pp_size: Pipeline stage count.
        dcp_size: Decode-context token-shard count.
        mla_only: Whether all cached KV is replicated within each DCP shard.

    Returns:
        One route per nonempty server/stage/token-shard intersection for MLA,
        or per nonempty server/stage intersection for worker-owned KV.

    Raises:
        ValueError: If the geometry cannot describe the vLLM rank layout.
    """
    n_servers = len(server_urls)
    if not n_servers or min(tp_size, pp_size, dcp_size) < 1:
        raise ValueError("server count, TP, PP and DCP must be positive")
    if len(set(server_urls)) != n_servers:
        raise ValueError("LMCache MP server URLs must be unique")
    world_size = tp_size * pp_size
    if world_size > 255:
        raise ValueError(
            "LMCache MP cache keys encode world size and rank in 8-bit fields; "
            f"TP × PP must not exceed 255 (got {world_size})"
        )
    if world_size % n_servers:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"n_servers ({n_servers})"
        )
    if tp_size % dcp_size:
        raise ValueError("tensor_parallel_size must be divisible by DCP size")

    ranks_per_server = world_size // n_servers
    routes: list[MPServerRoute] = []
    for server_index, server_url in enumerate(server_urls):
        server_start = server_index * ranks_per_server
        server_end = server_start + ranks_per_server
        for pp_rank in range(pp_size):
            stage_start = pp_rank * tp_size
            stage_end = stage_start + tp_size
            local_ranks = tuple(
                range(max(server_start, stage_start), min(server_end, stage_end))
            )
            if not local_ranks:
                continue
            dcp_ranks: tuple[int | None, ...] = (
                tuple(range(dcp_size)) if mla_only else (None,)
            )
            for dcp_rank in dcp_ranks:
                ranks = (
                    tuple(
                        rank
                        for rank in local_ranks
                        if (rank % tp_size) % dcp_size == dcp_rank
                    )
                    if dcp_rank is not None
                    else local_ranks
                )
                if not ranks:
                    continue
                kv_worker_ids = (
                    (pp_rank * dcp_size + dcp_rank,)
                    if dcp_rank is not None
                    else ranks
                )
                route_id = (
                    f"server{server_index}:pp{pp_rank}:dcp{dcp_rank}"
                    if dcp_rank is not None
                    else f"server{server_index}:pp{pp_rank}"
                )
                routes.append(
                    MPServerRoute(
                        route_id=route_id,
                        server_url=server_url,
                        server_index=server_index,
                        pp_rank=pp_rank,
                        dcp_rank=dcp_rank,
                        ranks=ranks,
                        kv_worker_ids=kv_worker_ids,
                        num_kv_readers=len(ranks) if mla_only else 1,
                    )
                )
    return tuple(routes)


def mp_stage_model_name(model_name: str, tp_size: int, pp_size: int, pp_rank: int) -> str:
    """Give a PP stage a stable storage and layout-registration namespace.

    Args:
        model_name: Existing LMCache model/cache namespace.
        tp_size: Tensor-parallel worker count.
        pp_size: Pipeline stage count.
        pp_rank: Pipeline stage of the registered layers.

    Returns:
        A name shared by equivalent workers on different MP servers.
    """
    if not 0 <= pp_rank < pp_size:
        raise ValueError(f"PP rank {pp_rank} is outside [0, {pp_size})")
    return f"{model_name}#mp-pp-v1-t{tp_size}-p{pp_size}-stage{pp_rank}"


def mp_route_request_id(
    request_id: str, pp_rank: int, dcp_rank: int | None
) -> str:
    """Qualify an engine request so server-local stage lookups cannot collide.

    Args:
        request_id: vLLM request id.
        pp_rank: Pipeline stage of this lookup domain.
        dcp_rank: Token shard when the MLA cache is shared by TP readers.

    Returns:
        An unambiguous session id used by scheduler and matching workers.
    """
    shard = "all" if dcp_rank is None else str(dcp_rank)
    return f"mp-pp-v1:{len(request_id)}:{request_id}:p{pp_rank}:d{shard}"
