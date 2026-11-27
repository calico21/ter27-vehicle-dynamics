"""Stage-2 closed-loop 14-DOF transient LTS runner & alpha_drv fastest-valid-lap finder."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .qss_solver import QSSResult
from .racing_line import RacingLineResult

LTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = LTS_DIR.parents[2]
SIM_DIR = REPO_ROOT / "src" / "mirena_sim"
RUNNER_GD = LTS_DIR / "lts_transient_runner.gd"


@dataclass
class TransientLapResult:
    """Result of closed-loop 14-DOF Godot headless run (1 flying lap or N continuous laps)."""

    label: str
    summary: Dict[str, Any]
    telemetry: Optional[pd.DataFrame]

    @property
    def valid(self) -> bool:
        return bool(self.summary.get("valid", False))

    @property
    def lap_time_s(self) -> float:
        return float(self.summary.get("mean_lap_time_s", 999.0))


def run_transient_lap(
    label: str,
    racing_line: RacingLineResult,
    qss: QSSResult,
    ecu_driving_mode: int = 2,
    tv_4wd_mode: int = 0,
    alpha_drv: float = 0.96,
    n_laps: int = 1,
    use_3d_kinematics: bool = True,
    mass_kg: Optional[float] = None,
    cl_a_scale: Optional[float] = None,
    p_eff_max_w: Optional[float] = None,
    mech_brake_start: Optional[float] = None,
    mode_overrides: Optional[Dict[str, Any]] = None,
    save_csv_path: Optional[Path] = None,
    grip_profile: str = "ttc_lab",
) -> TransientLapResult:
    """Run closed-loop 400 Hz Full14DofModel simulation in Godot headless."""
    base = racing_line.base_track
    race = racing_line.racing_track

    with tempfile.TemporaryDirectory(prefix="mirena_lts_") as tmp:
        tmp_dir = Path(tmp)
        job_path = tmp_dir / "lts_job.json"
        out_summary_path = tmp_dir / "lts_summary.json"
        csv_target = save_csv_path if save_csv_path is not None else (tmp_dir / "lts_trace.csv")
        if save_csv_path is not None:
            save_csv_path.parent.mkdir(parents=True, exist_ok=True)

        job: Dict[str, Any] = {
            "label": label,
            "ecu_driving_mode": int(ecu_driving_mode),
            "tv_4wd_mode": int(tv_4wd_mode),
            "use_3d_kinematics": bool(use_3d_kinematics),
            "alpha_drv": float(alpha_drv),
            "n_laps": int(n_laps),
            "csv_path": str(csv_target) if (save_csv_path is not None) else "",
            "base_track": {
                "length": float(base.length),
                "ds": float(base.ds),
                "s": [round(float(v), 4) for v in base.s],
                "x": [round(float(v), 5) for v in base.x],
                "y": [round(float(v), 5) for v in base.y],
                "psi": [round(float(v), 6) for v in base.psi],
                "w_left": [round(float(v), 4) for v in base.w_left],
                "w_right": [round(float(v), 4) for v in base.w_right],
            },
            "racing_track": {
                "length": float(race.length),
                "ds": float(race.ds),
                "s": [round(float(v), 4) for v in race.s],
                "x": [round(float(v), 5) for v in race.x],
                "y": [round(float(v), 5) for v in race.y],
                "psi": [round(float(v), 6) for v in race.psi],
                "kappa": [round(float(v), 6) for v in race.kappa],
            },
            "v_qss": [round(float(v), 4) for v in qss.v_qss],
            "ax_qss": [round(float(a), 4) for a in qss.ax_qss],
        }
        if mass_kg is not None:
            job["mass_kg"] = float(mass_kg)
        if cl_a_scale is not None:
            job["cl_a_scale"] = float(cl_a_scale)
        if p_eff_max_w is not None:
            job["p_eff_max_w"] = float(p_eff_max_w)
        if mech_brake_start is not None:
            job["mech_brake_start"] = float(mech_brake_start)
        if mode_overrides is not None:
            job["mode_overrides"] = mode_overrides
        if grip_profile and grip_profile != "ttc_lab":
            job["grip_profile"] = str(grip_profile)

        job_path.write_text(json.dumps(job), encoding="utf-8")
        cmd = [
            "godot",
            "--headless",
            "--path",
            str(SIM_DIR),
            "-s",
            str(RUNNER_GD),
            "--",
            str(job_path),
            str(out_summary_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out_summary_path.exists():
            raise RuntimeError(
                f"Godot LTS runner failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        summary = json.loads(out_summary_path.read_text(encoding="utf-8"))
        df = pd.read_csv(csv_target) if (save_csv_path is not None and csv_target.exists()) else None
        return TransientLapResult(label=label, summary=summary, telemetry=df)


def search_fastest_valid_lap(
    label: str,
    racing_line: RacingLineResult,
    qss: QSSResult,
    ecu_driving_mode: int = 2,
    tv_4wd_mode: int = 0,
    n_laps: int = 1,
    use_3d_kinematics: bool = True,
    mass_kg: Optional[float] = None,
    cl_a_scale: Optional[float] = None,
    p_eff_max_w: Optional[float] = None,
    mech_brake_start: Optional[float] = None,
    mode_overrides: Optional[Dict[str, Any]] = None,
    save_csv_path: Optional[Path] = None,
    alpha_bounds: Tuple[float, float] = (0.90, 1.02),
    grip_profile: str = "ttc_lab",
) -> Tuple[TransientLapResult, List[Dict[str, Any]]]:
    """
    Lightweight bisection + coarse search (4-5 evaluations) over alpha_drv in [0.90, 1.02]
    to find the fastest strictly valid closed-loop lap, then logs final CSV for that optimum.
    """
    lo, hi = float(alpha_bounds[0]), float(alpha_bounds[1])
    history: List[Dict[str, Any]] = []
    cache: Dict[float, TransientLapResult] = {}

    def eval_alpha(a: float, csv_out: Optional[Path] = None) -> TransientLapResult:
        ar = round(a, 4)
        if csv_out is None and ar in cache:
            return cache[ar]
        r = run_transient_lap(
            label=label,
            racing_line=racing_line,
            qss=qss,
            ecu_driving_mode=ecu_driving_mode,
            tv_4wd_mode=tv_4wd_mode,
            alpha_drv=ar,
            n_laps=1,  # search on single lap, final multi-lap run logs n_laps
            use_3d_kinematics=use_3d_kinematics,
            mass_kg=mass_kg,
            cl_a_scale=cl_a_scale,
            p_eff_max_w=p_eff_max_w,
            mech_brake_start=mech_brake_start,
            mode_overrides=mode_overrides,
            save_csv_path=csv_out,
            grip_profile=grip_profile,
        )
        cache[ar] = r
        history.append({
            "alpha": ar,
            "valid": r.valid,
            "lap_time_s": r.lap_time_s,
            "off_track": int(r.summary.get("off_track_ticks", 0)),
            "max_beta_deg": float(r.summary.get("max_abs_beta_deg", 0.0)),
            "max_ey_base_m": float(r.summary.get("max_abs_ey_base_m", 0.0)),
        })
        return r

    # 1. Probe 0.96 midpoint first, then bisect [lo, hi] in 4 steps to locate validity/speed frontier
    r_mid = eval_alpha(0.5 * (lo + hi))
    best_valid: Optional[TransientLapResult] = r_mid if r_mid.valid else None

    if r_mid.valid:
        a_lo = 0.5 * (lo + hi)
        a_hi = hi
    else:
        a_lo = lo
        a_hi = 0.5 * (lo + hi)
        r_lo = eval_alpha(a_lo)
        if r_lo.valid:
            best_valid = r_lo

    for _ in range(3):
        a_try = 0.5 * (a_lo + a_hi)
        r_try = eval_alpha(a_try)
        if r_try.valid:
            if best_valid is None or r_try.lap_time_s < best_valid.lap_time_s:
                best_valid = r_try
            a_lo = a_try
        else:
            a_hi = a_try

    if best_valid is None:
        for a_fb in (0.86, 0.82):
            r_fb = eval_alpha(a_fb)
            if r_fb.valid:
                best_valid = r_fb
                break

    best_alpha = float(best_valid.summary["alpha_drv"]) if best_valid is not None else lo
    if n_laps == 1 and save_csv_path is None and best_valid is not None:
        return best_valid, history

    # Final run with exact optimal alpha_drv logging full telemetry (and n_laps if > 1)
    final_res = run_transient_lap(
        label=label,
        racing_line=racing_line,
        qss=qss,
        ecu_driving_mode=ecu_driving_mode,
        tv_4wd_mode=tv_4wd_mode,
        alpha_drv=best_alpha,
        n_laps=n_laps,
        use_3d_kinematics=use_3d_kinematics,
        mass_kg=mass_kg,
        cl_a_scale=cl_a_scale,
        p_eff_max_w=p_eff_max_w,
        mech_brake_start=mech_brake_start,
        mode_overrides=mode_overrides,
        save_csv_path=save_csv_path,
        grip_profile=grip_profile,
    )
    final_res.summary["alpha_search_history"] = history
    return final_res, history
