# SPDX-License-Identifier: Apache-2.0
"""Measure first-request TTFT and MP CUDA IPC registration timings.

The target vLLM instance and LMCache MP server must already be running. This
driver sends OpenAI-compatible streaming chat requests and can summarize the
registration timing log lines emitted by LMCache's client and server.
"""

# Standard
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import secrets
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class RequestSample:
    """Latency measurements for one streaming completion request."""

    request_index: int
    ttft_ms: float
    total_ms: float


CLIENT_REGISTRATION_RE = re.compile(
    r"MP CUDA IPC client registration round trip completed: "
    r"instance_id=(?P<instance_id>\d+) latency_ms=(?P<latency_ms>[\d.]+) "
    r"mq_timeout_s=(?P<timeout_s>[\d.]+)"
)
METADATA_REGISTRATION_RE = re.compile(
    r"MP KV cache metadata registration completed: "
    r"instance_id=(?P<instance_id>\d+) latency_ms=(?P<latency_ms>[\d.]+)"
)
GATE_RE = re.compile(
    r"MP CUDA IPC registration gate completed: "
    r"instance_id=(?P<instance_id>\d+) trigger=(?P<trigger>\S+) "
    r"latency_ms=(?P<latency_ms>[\d.]+) "
    r"mq_timeout_s=(?P<timeout_s>[\d.]+)"
)
SERVER_REGISTRATION_RE = re.compile(
    r"MP server CUDA IPC registration completed: "
    r"instance_id=(?P<instance_id>\d+) .*latency_ms=(?P<latency_ms>[\d.]+)"
)


def _read_prompt(prompt_file: Path) -> str:
    """Read a non-empty UTF-8 prompt from ``prompt_file``."""
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompt


def _unique_store_prompt(seed: str, request_index: int) -> str:
    """Return a unique prompt with approximately the same word count."""
    words = seed.split()
    nonce = secrets.token_hex(8)
    return " ".join(
        f"{secrets.token_hex(4)}-{nonce}-{request_index}-{index}"
        for index, _ in enumerate(words)
    )


def _authorization_headers() -> dict[str, str]:
    """Return optional bearer authentication from ``OPENAI_API_KEY``."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def _get_model(endpoint: str, timeout_s: float) -> str:
    """Read the first model id from the endpoint's OpenAI models route."""
    endpoint_parts = endpoint.rstrip("/").split("/")
    models_url = "/".join(endpoint_parts[:-2]) + "/models"
    request = Request(models_url, headers=_authorization_headers())
    with urlopen(request, timeout=timeout_s) as response:
        body = json.loads(response.read())
    models = body.get("data", [])
    if not models or not models[0].get("id"):
        raise ValueError(f"No model id returned by {models_url}")
    return str(models[0]["id"])


def _measure_request(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    request_index: int,
) -> RequestSample:
    """Send one streaming chat request and measure TTFT and total latency."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0,
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **_authorization_headers(),
    }
    request = Request(endpoint, data=payload, headers=headers, method="POST")
    started_at = time.perf_counter()
    first_token_at: float | None = None

    try:
        with urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach vLLM endpoint {endpoint}: {exc}") from exc

    finished_at = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError(
            f"Request {request_index} completed without a streamed text token"
        )
    return RequestSample(
        request_index=request_index,
        ttft_ms=(first_token_at - started_at) * 1000,
        total_ms=(finished_at - started_at) * 1000,
    )


def _percentile(values: list[float], percentile: float) -> float:
    """Return the nearest-rank percentile of non-empty ``values``."""
    ordered = sorted(values)
    index = max(0, int((percentile / 100) * len(ordered) + 0.999999) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _summarize_samples(
    samples: list[RequestSample],
) -> dict[str, float | int] | None:
    """Summarize a set of request samples, or return None when empty."""
    if not samples:
        return None
    ttft_values = [sample.ttft_ms for sample in samples]
    total_values = [sample.total_ms for sample in samples]
    return {
        "count": len(samples),
        "ttft_p50_ms": statistics.median(ttft_values),
        "ttft_p95_ms": _percentile(ttft_values, 95),
        "ttft_max_ms": max(ttft_values),
        "total_p50_ms": statistics.median(total_values),
        "total_p95_ms": _percentile(total_values, 95),
    }


def _read_registration_log(
    log_path: Path, pattern: re.Pattern[str]
) -> list[dict[str, Any]]:
    """Extract registration timing values from one process log."""
    values: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match is not None:
            values.append(match.groupdict())
    return values


def _summarize_registration_logs(
    client_log: Path | None,
    server_log: Path | None,
    mq_timeout_s: float,
    expected_trigger: str,
) -> dict[str, Any]:
    """Summarize client, server, and first-operation gate log timings."""
    summary: dict[str, Any] = {}
    if client_log is not None:
        client = _read_registration_log(client_log, CLIENT_REGISTRATION_RE)
        metadata = _read_registration_log(client_log, METADATA_REGISTRATION_RE)
        gates = _read_registration_log(client_log, GATE_RE)
        gates = [
            row
            for row in gates
            if row["trigger"].upper() == expected_trigger.upper()
        ]
        client_instances = {row["instance_id"] for row in client}
        summary["client_full_registration"] = _latency_summary(client)
        summary["client_metadata_registration"] = _latency_summary(metadata)
        summary["first_operation_registration_gate"] = _latency_summary(gates)
        summary["mq_timeout_s"] = mq_timeout_s
    if server_log is not None:
        server = _read_registration_log(server_log, SERVER_REGISTRATION_RE)
        if client_log is not None:
            server = [row for row in server if row["instance_id"] in client_instances]
        summary["server_full_registration"] = _latency_summary(server)
    timeout_ms = mq_timeout_s * 1000
    timeout_fractions = {
        key: timing["max_ms"] / timeout_ms
        for key, timing in summary.items()
        if isinstance(timing, dict)
        and timing.get("max_ms") is not None
        and key != "client_metadata_registration"
    }
    if timeout_fractions:
        summary["max_timeout_fractions"] = timeout_fractions
    return summary


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-worker ``latency_ms`` values from log rows."""
    values = [float(row["latency_ms"]) for row in rows]
    return {
        "samples": len(values),
        "latency_ms": values,
        "max_ms": max(values) if values else None,
    }


def _parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1/chat/completions",
        help="OpenAI-compatible vLLM chat completions endpoint",
    )
    parser.add_argument("--model", help="Model id; defaults to the first /v1/models id")
    parser.add_argument(
        "--mode",
        choices=("store", "retrieve"),
        required=True,
        help="Expected first LMCache transfer operation for this fresh worker",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Representative prompt; retrieve mode must match a pre-seeded prompt",
    )
    parser.add_argument(
        "--requests", type=int, default=5, help="Total requests; first is measured cold"
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--mq-timeout", type=float, default=300.0)
    parser.add_argument("--client-log", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.timeout <= 0 or args.mq_timeout <= 0:
        parser.error("--timeout and --mq-timeout must be positive")
    if args.mode == "retrieve" and not args.prompt_file.is_file():
        parser.error("--prompt-file must exist and contain the seeded prompt")
    return args


def main() -> None:
    """Run the benchmark and print machine-readable latency results."""
    args = _parse_args()
    seed_prompt = _read_prompt(args.prompt_file)
    model = args.model or _get_model(args.endpoint, args.timeout)

    samples: list[RequestSample] = []
    for request_index in range(args.requests):
        prompt = (
            _unique_store_prompt(seed_prompt, request_index)
            if args.mode == "store"
            else seed_prompt
        )
        samples.append(
            _measure_request(
                args.endpoint,
                model,
                prompt,
                args.max_tokens,
                args.timeout,
                request_index,
            )
        )

    first_sample = samples[0]
    steady_summary = _summarize_samples(samples[1:])
    result = {
        "mode": args.mode,
        "model": model,
        "endpoint": args.endpoint,
        "request_count": len(samples),
        "first_operation": asdict(first_sample),
        "steady_state": steady_summary,
        "first_request_ttft_delta_vs_steady_p50_ms": (
            first_sample.ttft_ms - steady_summary["ttft_p50_ms"]
            if steady_summary is not None
            else None
        ),
        "registration_logs": _summarize_registration_logs(
            args.client_log,
            args.server_log,
            args.mq_timeout,
            args.mode,
        ),
    }
    result_json = json.dumps(result, indent=2, sort_keys=True)
    print(result_json)
    if args.output is not None:
        args.output.write_text(result_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
