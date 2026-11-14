"""Shared headless Godot binary discovery, JSON result parsing, and parallel execution helpers."""

from __future__ import annotations

import concurrent.futures
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable


def find_godot_binary(requested: str | None = None) -> str:
    """Resolve Godot binary path from CLI flag, PATH, or standard Antigravity install."""
    if requested:
        return requested
    found = shutil.which("godot") or shutil.which("godot4")
    if found:
        return found
    fallback = Path.home() / ".gemini" / "antigravity" / "bin" / "godot"
    if fallback.exists():
        return str(fallback)
    return "godot"


def extract_result_json(stdout_text: str, prefix: str) -> dict[str, Any] | None:
    """Parse `<prefix>{...}` JSON line emitted by a headless Godot evaluation script."""
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            payload = stripped[len(prefix) :].strip()
            return json.loads(payload)
    return None


def run_parallel_jobs(
    worker_fn: Callable[[Any], dict[str, Any]],
    jobs: Iterable[Any],
    max_workers: int,
) -> list[dict[str, Any]]:
    """Execute evaluation jobs in parallel via ProcessPoolExecutor and return results."""
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(worker_fn, jobs):
            results.append(res)
    return results
