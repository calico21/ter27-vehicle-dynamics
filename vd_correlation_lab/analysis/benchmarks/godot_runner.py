#!/usr/bin/env python3
"""
Orquestador de ejecución headless en Godot (src/mirena_sim/tools/vd_lab_replay.gd).
Ejecuta lotes en paralelo (ThreadPoolExecutor) para acelerar replays multi-tanda.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


def _run_single_job(repo_root: Path, job: Dict[str, Any], timeout_s: int) -> None:
    sim_dir = repo_root / "src" / "mirena_sim"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump([job], tf, indent=2)
        batch_path = Path(tf.name)
    try:
        replay_script = repo_root / "vd_correlation_lab" / "analysis" / "benchmarks" / "vd_lab_replay.gd"
        cmd = [
            "godot", "--headless", "--path", str(sim_dir),
            "-s", str(replay_script),
            "--", "--batch", str(batch_path),
        ]
        res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout_s)
        if res.returncode != 0:
            raise RuntimeError(
                f"Godot vd_lab_replay.gd falló (código {res.returncode}):\n{res.stderr}\n{res.stdout}"
            )
    finally:
        batch_path.unlink(missing_ok=True)


def run_godot_batch(repo_root: Path, jobs: List[Dict[str, Any]], timeout_s: int = 240, max_workers: int = 4) -> None:
    """Ejecuta un lote de simulaciones en paralelo en src/mirena_sim."""
    if not jobs:
        return
    if len(jobs) == 1:
        _run_single_job(repo_root, jobs[0], timeout_s)
        return
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        futs = {pool.submit(_run_single_job, repo_root, job, timeout_s): idx for idx, job in enumerate(jobs)}
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            print(f"  [Godot] {done}/{len(jobs)} simulaciones completadas", flush=True)


def ensure_synthetic_log(repo_root: Path, logs_dir: Path, vehicle_id: str, calib: Dict[str, Any]) -> Path:
    """Genera un log sintético de maniobra combinada para vehículos sin logs CAN de pista."""
    out_csv = logs_dir / f"{vehicle_id}_synthetic_track.csv"
    jobs = [{
        "kind": "synthetic",
        "vehicle_id": vehicle_id,
        "out_csv": str(out_csv),
        "duration_s": 40.0,
        "ecu_mode": int(calib.get("ecu_mode", 2)),
        "tv_map_id": int(calib.get("tv_map_id", 2)),
    }]
    run_godot_batch(repo_root, jobs, timeout_s=120)
    return out_csv
