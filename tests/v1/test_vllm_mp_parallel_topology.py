# SPDX-License-Identifier: Apache-2.0
"""Rank routing tests for multi-server vLLM TP/PP topologies."""

from __future__ import annotations

import pytest

from lmcache.integration.vllm.mp_parallel_topology import (
    build_mp_server_routes,
    mp_route_request_id,
    mp_stage_model_name,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy


@pytest.mark.parametrize(
    "tp_size,pp_size,n_servers",
    [
        (1, 2, 2),  # one LMCache server per PP stage
        (2, 2, 2),  # one TP group per server
        (2, 3, 3),  # multiple stages and multiple TP ranks per server
        (4, 2, 4),  # one server per worker, with two servers per stage
    ],
)
def test_worker_owned_kv_routes_cover_each_rank_once(
    tp_size: int, pp_size: int, n_servers: int
) -> None:
    """Every PP/TP worker is assigned to exactly one server and stage."""
    urls = [f"tcp://server-{i}:5555" for i in range(n_servers)]
    routes = build_mp_server_routes(
        urls, tp_size, pp_size, dcp_size=1, mla_only=False
    )

    assert sorted(rank for route in routes for rank in route.ranks) == list(
        range(tp_size * pp_size)
    )
    assert len({rank for route in routes for rank in route.ranks}) == tp_size * pp_size
    for route in routes:
        assert route.server_url == urls[route.server_index]
        assert all(rank // tp_size == route.pp_rank for rank in route.ranks)
        assert route.kv_worker_ids == route.ranks
        assert route.num_kv_readers == 1


def test_one_server_per_pp_stage_routes_tp_group_as_one_mla_shard() -> None:
    """Replicated MLA has one shared object per stage and DCP shard."""
    routes = build_mp_server_routes(
        ["server-0", "server-1"],
        tp_size=4,
        pp_size=2,
        dcp_size=2,
        mla_only=True,
    )

    observed = [
        (
            route.server_index,
            route.pp_rank,
            route.dcp_rank,
            route.ranks,
            route.kv_worker_ids,
            route.num_kv_readers,
        )
        for route in routes
    ]
    assert observed == [
        (0, 0, 0, (0, 2), (0,), 2),
        (0, 0, 1, (1, 3), (1,), 2),
        (1, 1, 0, (4, 6), (2,), 2),
        (1, 1, 1, (5, 7), (3,), 2),
    ]


@pytest.mark.parametrize(
    "tp_size,pp_size,n_servers,dcp_size",
    [(2, 2, 2, 1), (4, 2, 2, 2), (4, 2, 4, 2)],
)
def test_partitioned_mla_routes_balance_readers_and_writers(
    tp_size: int, pp_size: int, n_servers: int, dcp_size: int
) -> None:
    """Each shared MLA shard has exact lookup locks and one store writer."""
    routes = build_mp_server_routes(
        [f"tcp://server-{i}:5555" for i in range(n_servers)],
        tp_size,
        pp_size,
        dcp_size,
        mla_only=True,
    )

    for route in routes:
        strategies = [
            ParallelStrategy(
                mla_only=True,
                vllm_world_size=tp_size * pp_size,
                vllm_worker_id=rank,
                tp_size=tp_size,
                pp_size=pp_size,
                n_servers=n_servers,
                dcp_size=dcp_size,
            )
            for rank in route.ranks
        ]

        assert [strategy.kv_worker_id for strategy in strategies] == [
            route.kv_worker_ids[0]
        ] * len(route.ranks)
        assert all(
            strategy.num_kv_readers == len(route.ranks)
            for strategy in strategies
        )
        assert sum(strategy.is_kv_writer for strategy in strategies) == 1


def test_partitioned_hybrid_routes_store_each_rank_piece_once() -> None:
    """Non-replicated hybrid KV uses one object and writer per worker rank."""
    routes = build_mp_server_routes(
        ["tcp://server-0:5555", "tcp://server-1:5555"],
        tp_size=2,
        pp_size=2,
        dcp_size=1,
        mla_only=False,
    )

    for route in routes:
        strategies = [
            ParallelStrategy(
                mla_only=False,
                vllm_world_size=4,
                vllm_worker_id=rank,
                tp_size=2,
                pp_size=2,
                n_servers=2,
            )
            for rank in route.ranks
        ]
        assert [strategy.kv_worker_id for strategy in strategies] == list(
            route.kv_worker_ids
        )
        assert all(strategy.num_kv_readers == 1 for strategy in strategies)
        assert all(strategy.is_kv_writer for strategy in strategies)


def test_servers_may_partition_tp_ranks_inside_each_pp_stage() -> None:
    """The topology remains complete when endpoints split each TP stage."""
    routes = build_mp_server_routes(
        [f"server-{i}" for i in range(4)],
        tp_size=4,
        pp_size=2,
        dcp_size=1,
        mla_only=True,
    )

    assert [(r.server_index, r.pp_rank, r.ranks) for r in routes] == [
        (0, 0, (0, 1)),
        (1, 0, (2, 3)),
        (2, 1, (4, 5)),
        (3, 1, (6, 7)),
    ]
    assert [r.num_kv_readers for r in routes] == [2, 2, 2, 2]


@pytest.mark.parametrize(
    "urls,tp_size,pp_size,dcp_size,mla_only,message",
    [
        ([], 1, 1, 1, False, "positive"),
        (["same", "same"], 1, 2, 1, False, "unique"),
        (["a", "b", "c"], 2, 2, 1, False, "divisible"),
        (["a"], 2, 2, 3, True, "divisible by DCP"),
    ],
)
def test_invalid_server_topologies_fail_fast(
    urls: list[str],
    tp_size: int,
    pp_size: int,
    dcp_size: int,
    mla_only: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_mp_server_routes(urls, tp_size, pp_size, dcp_size, mla_only)


def test_pp_stage_cache_names_and_request_ids_are_isolated() -> None:
    """Stage namespaces and sessions cannot collide across PP endpoints."""
    assert mp_stage_model_name("model", 2, 2, 0) != mp_stage_model_name(
        "model", 2, 2, 1
    )
    assert mp_route_request_id("req-1", 0, None) != mp_route_request_id(
        "req-1", 1, None
    )
    assert mp_route_request_id("a:p1", 0, None) != mp_route_request_id(
        "a", 1, None
    )
