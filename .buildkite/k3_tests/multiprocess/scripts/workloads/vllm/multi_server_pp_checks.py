# SPDX-License-Identifier: Apache-2.0
"""HTTP assertions for the two-server PP correctness workload."""

# Standard
from collections.abc import Callable
from pathlib import Path
from typing import Any
import argparse
import json
import time
import urllib.error
import urllib.request


def read_json(url: str) -> dict[str, Any]:
    """Read a local test endpoint, rejecting non-object JSON responses.

    Args:
        url: HTTP endpoint of a test-owned process.

    Returns:
        Its JSON object response.

    Raises:
        ValueError: If the endpoint returned a non-object payload.
        urllib.error.URLError: If the endpoint is unreachable.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return result


def _is_idle(status: dict[str, Any]) -> bool:
    """Include transfers and L2 stores so a later clear cannot race their writes."""
    storage = status["storage_manager"]
    return (
        status["active_sessions"] == 0
        and status["active_prefetch_jobs"] == 0
        and all(
            l1["read_locked_count"] == 0 and l1["write_locked_count"] == 0
            for l1 in storage["l1_managers"].values()
        )
        and all(
            store["pending_keys_count"] == 0 and store["in_flight_task_count"] == 0
            for store in storage["store_controllers"].values()
        )
    )


def wait_status(
    port: int, predicate: Callable[[dict[str, Any]], bool], timeout: int
) -> None:
    """Wait for an MP HTTP status predicate within a bounded deadline.

    Args:
        port: Test server's HTTP port.
        predicate: Required property of its status response.
        timeout: Maximum seconds to wait.

    Raises:
        RuntimeError: If the predicate is never satisfied.
    """
    deadline = time.monotonic() + timeout
    last_status: object = "not queried"
    while time.monotonic() < deadline:
        try:
            last_status = read_json(f"http://127.0.0.1:{port}/status")
            if predicate(last_status):
                return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_status = str(exc)
        time.sleep(1)
    raise RuntimeError(f"MP server {port} did not settle: {last_status}")


def cancel_stream(port: int, model: str, prompt: Path) -> None:
    """Disconnect after the first generated token of a long streaming request.

    Args:
        port: vLLM's HTTP port.
        model: Served model ID.
        prompt: Prompt file from the cold/cache-hit comparison.

    Raises:
        RuntimeError: If no generated text arrived before the stream ended.
    """
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt.read_text(),
            "temperature": 0,
            "max_tokens": 512,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            result = json.loads(payload)
            choices = result.get("choices", [])
            if choices and choices[0].get("text"):
                print("Disconnecting a live stream after its first generated text")
                return
    raise RuntimeError("The cancellation test never observed a live generation")


def main() -> None:
    """Run an idle, registration, empty-cache, or stream-disconnect assertion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=["idle", "registered", "empty", "cancel"])
    parser.add_argument("ports", nargs="+", type=int)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args()
    if args.check == "cancel":
        if not args.model or args.prompt is None:
            parser.error("cancel requires --model and --prompt")
        cancel_stream(args.ports[0], args.model, args.prompt)
    else:
        for port in args.ports:
            if args.check == "empty":
                wait_status(
                    port,
                    lambda s: all(
                        l1["total_object_count"] == 0
                        and l1["read_locked_count"] == 0
                        and l1["write_locked_count"] == 0
                        for l1 in s["storage_manager"]["l1_managers"].values()
                    ),
                    args.timeout,
                )
            elif args.check == "registered":
                wait_status(port, lambda s: bool(s["cache_context_meta"]), args.timeout)
            else:
                wait_status(port, _is_idle, args.timeout)
    print(f"PASS: {args.check} on {args.ports}")


if __name__ == "__main__":
    main()
