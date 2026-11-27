"""Asynchronous CasADi lap-time and dynamic event simulation runner."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from api.core.config import (
    PYTHON_BIN,
    REPO_ROOT,
    SETUP_REPORT_PATH,
    SOLVE_DYNAMIC_SCRIPT,
    SOLVE_FULL_LAP_SCRIPT,
)
from api.models.simulation import SimulationJobStatusResponse, SimulationRunRequest


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SimulationRunner:
    """Manages asynchronous CasADi IPOPT solver jobs and cached dry-run evaluations."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def _build_solver_command(self, req: SimulationRunRequest) -> List[str]:
        if req.event in ("skidpad", "accel"):
            cmd = [
                PYTHON_BIN,
                "-u",
                str(SOLVE_DYNAMIC_SCRIPT),
                "--event",
                req.event,
                "--grip-profile",
                req.grip_profile,
                "--compound",
                req.compound,
            ]
            if req.optimize_setup:
                cmd.append("--optimize-setup")
            return cmd

        cmd = [
            PYTHON_BIN,
            "-u",
            str(SOLVE_FULL_LAP_SCRIPT),
            "--mode",
            req.mode,
            "--compound",
            req.compound,
            "--n-intervals",
            str(req.n_intervals),
            "--max-iter",
            str(req.max_iter),
            "--print-level",
            "0",
        ]
        if req.event == "endurance":
            cmd.extend(["--energy-budget-kwh", "3.5"])
        if req.optimize_setup:
            cmd.append("--optimize-setup")
        return cmd

    @staticmethod
    def _load_cached_event_result(event: str) -> Dict[str, Any]:
        if not SETUP_REPORT_PATH.exists():
            return {"event": event, "source": "dry_run_fallback"}
        rep = json.loads(SETUP_REPORT_PATH.read_text(encoding="utf-8"))
        ev_data = rep.get("events", {}).get(event, {})
        return {
            "event": event,
            "source": "trackside_setup_optimization_report.json",
            "metrics": ev_data,
            "fs_points": rep.get("fs_points", {}),
        }

    async def _run_job_subprocess(self, job_id: str, cmd: List[str], event: str) -> None:
        job = self._jobs[job_id]
        job["status"] = "running"
        job["started_at_utc"] = _utc_now()
        job["progress_pct"] = 10.0

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            log_lines: List[str] = []
            assert proc.stdout is not None
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                if line:
                    log_lines.append(line)
                    if len(log_lines) > 40:
                        log_lines = log_lines[-40:]
                    job["log_tail"] = list(log_lines)
                    job["progress_pct"] = min(92.0, job["progress_pct"] + 2.5)

            rc = await proc.wait()
            job["finished_at_utc"] = _utc_now()
            if rc == 0:
                job["status"] = "completed"
                job["progress_pct"] = 100.0
                job["result"] = self._load_cached_event_result(event)
            else:
                job["status"] = "failed"
                job["error"] = f"Solver exited with return code {rc}"
        except Exception as exc:
            job["finished_at_utc"] = _utc_now()
            job["status"] = "failed"
            job["error"] = str(exc)

    async def start_job(self, req: SimulationRunRequest) -> SimulationJobStatusResponse:
        job_id = f"sim_{req.event}_{uuid.uuid4().hex[:8]}"
        cmd = self._build_solver_command(req)
        now = _utc_now()

        job_record: Dict[str, Any] = {
            "job_id": job_id,
            "event": req.event,
            "grip_profile": req.grip_profile,
            "compound": req.compound,
            "optimize_setup": req.optimize_setup,
            "status": "queued",
            "progress_pct": 0.0,
            "created_at_utc": now,
            "started_at_utc": None,
            "finished_at_utc": None,
            "command": cmd,
            "result": None,
            "error": None,
            "log_tail": [],
        }
        self._jobs[job_id] = job_record

        if req.dry_run:
            job_record["status"] = "completed"
            job_record["started_at_utc"] = now
            job_record["finished_at_utc"] = now
            job_record["progress_pct"] = 100.0
            job_record["result"] = self._load_cached_event_result(req.event)
            job_record["log_tail"] = [
                f"dry_run=True: returning validated CasADi reference metrics for {req.event}"
            ]
        else:
            asyncio.create_task(self._run_job_subprocess(job_id, cmd, req.event))

        return SimulationJobStatusResponse.model_validate(job_record)

    def get_job_status(self, job_id: str) -> Optional[SimulationJobStatusResponse]:
        record = self._jobs.get(job_id)
        if record is None:
            return None
        return SimulationJobStatusResponse.model_validate(record)


simulation_runner = SimulationRunner()
