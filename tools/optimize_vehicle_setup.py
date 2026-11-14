#!/usr/bin/env python3
"""Multi-objective mechanical setup & brake-bias optimizer for Mirena (TeR27).

Runs parallel 14-DOF closed-loop simulations via headless Godot
(res://tests/eval_setup_candidate.gd) across 3 maneuvers (Skidpad, Slalom,
Trail Braking) using:
  Phase 1: Stratified Latin Hypercube Sampling (LHS) exploration (+ Baseline)
  Phase 2: Surrogate modeling (SciPy RBFInterpolator / NumPy RBF fallback)
           and multi-start optimization for 3 event objectives
  Phase 3: Real headless Godot validation and preset JSON export

Usage:
  python3 tools/optimize_vehicle_setup.py --budget 24 --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _godot_runner import extract_result_json, find_godot_binary

import numpy as np

try:
    from scipy.interpolate import RBFInterpolator  # type: ignore
    from scipy.optimize import minimize as scipy_minimize  # type: ignore
    from scipy.stats import qmc  # type: ignore

    HAS_SCIPY = True
except Exception:
    RBFInterpolator = None
    scipy_minimize = None
    qmc = None
    HAS_SCIPY = False


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_DIR = REPO_ROOT / "src" / "mirena_sim"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_DIR / "Scenes" / "Vehicle" / "Dynamics" / "Core" / "Config" / "setup_presets"
)
EVAL_SCRIPT_RES = "res://tests/eval_setup_candidate.gd"
RESULT_PREFIX = "SETUP_OPT_RESULT:"

PARAM_NAMES: tuple[str, ...] = (
    "spring_stiffness_front",
    "spring_stiffness_rear",
    "arb_stiffness_front",
    "arb_stiffness_rear",
    "static_camber_front_deg",
    "static_camber_rear_deg",
    "toe_front_deg",
    "toe_rear_deg",
    "brake_bias_front",
    "aero_balance_front",
)

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "spring_stiffness_front": (30000.0, 65000.0),
    "spring_stiffness_rear": (30000.0, 65000.0),
    "arb_stiffness_front": (1500.0, 9000.0),
    "arb_stiffness_rear": (1500.0, 9000.0),
    "static_camber_front_deg": (-3.0, -0.8),
    "static_camber_rear_deg": (-2.5, -0.5),
    "toe_front_deg": (-0.5, 0.3),
    "toe_rear_deg": (-0.2, 0.5),
    "brake_bias_front": (0.54, 0.66),
    "aero_balance_front": (0.44, 0.52),
}

PARAM_PRECISION: dict[str, int] = {
    "spring_stiffness_front": 0,
    "spring_stiffness_rear": 0,
    "arb_stiffness_front": 0,
    "arb_stiffness_rear": 0,
    "static_camber_front_deg": 2,
    "static_camber_rear_deg": 2,
    "toe_front_deg": 3,
    "toe_rear_deg": 3,
    "brake_bias_front": 4,
    "aero_balance_front": 4,
}

# Physical baseline defaults matching VehicleConfig + Config/*.json
BASELINE_PARAMS: dict[str, float] = {
    "spring_stiffness_front": 35000.0,
    "spring_stiffness_rear": 48000.0,
    "arb_stiffness_front": 4500.0,
    "arb_stiffness_rear": 5500.0,
    "static_camber_front_deg": -2.0,
    "static_camber_rear_deg": -1.5,
    "toe_front_deg": 0.0,
    "toe_rear_deg": 0.0,
    "brake_bias_front": 0.58,
    "aero_balance_front": 0.48,
}

SURROGATE_METRICS: tuple[str, ...] = (
    "skidpad_lap_time_s",
    "skidpad_lat_g",
    "skidpad_wz_rmse",
    "skidpad_peak_beta_deg",
    "slalom_wz_rmse_rads",
    "slalom_agility_s",
    "trail_brake_time_s",
    "trail_brake_dist_m",
    "high_speed_beta_max_deg",
    "high_speed_wz_rmse",
    "brake_lockup_peak",
    "efficiency_kwh_km",
    "min_fz_n",
    "max_roll_deg",
    "peak_beta_deg",
    "rear_lock_margin",
)

EVENT_SPECS: dict[str, dict[str, str]] = {
    "opt_skidpad_setup": {
        "name": "Opt Skidpad Setup",
        "description": "Surrogate-optimized mechanical & aero setup for minimum 9.125 m skidpad lap time and yaw precision.",
    },
    "opt_autocross_setup": {
        "name": "Opt Autocross Setup",
        "description": "Balanced setup optimizing skidpad grip, 65 km/h slalom yaw agility, trail-braking response, and stability.",
    },
    "opt_endurance_setup": {
        "name": "Opt Endurance Setup",
        "description": "Efficiency and thermal/sideslip stability setup minimizing kWh/km and peak sideslip while preserving pace.",
    },
}


@dataclass(frozen=True)
class EvalJob:
    index: int
    label: str
    params: dict[str, float]
    project_dir: str
    godot_bin: str
    map_id: int
    timeout_s: float


def resolve_godot_bin(requested: str | None) -> str:
    """Resolve Godot binary path from CLI flag, PATH, or standard install."""
    return find_godot_binary(requested)


def load_baseline_from_config(project_dir: Path) -> dict[str, float]:
    """Read current baseline values from Config/*.json with safe defaults."""
    base = dict(BASELINE_PARAMS)
    cfg_dir = project_dir / "Scenes" / "Vehicle" / "Dynamics" / "Core" / "Config"
    for fname in ("suspension.json", "brakes.json", "aero.json"):
        fpath = cfg_dir / fname
        if fpath.is_file():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key in PARAM_NAMES:
                        if key in data and isinstance(data[key], (int, float)):
                            base[key] = float(data[key])
            except Exception:
                pass
    return base


def params_to_unit(params: dict[str, float]) -> np.ndarray:
    """Normalize parameter dictionary into [0, 1]^D."""
    vec = np.empty(len(PARAM_NAMES), dtype=np.float64)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        vec[i] = (float(params[name]) - lo) / max(hi - lo, 1e-12)
    return np.clip(vec, 0.0, 1.0)


def unit_to_params(u: np.ndarray) -> dict[str, float]:
    """Map normalized [0, 1]^D vector back to rounded physical setup parameters."""
    u_clipped = np.clip(u, 0.0, 1.0)
    out: dict[str, float] = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        val = lo + float(u_clipped[i]) * (hi - lo)
        prec = PARAM_PRECISION[name]
        out[name] = round(val, prec) if prec > 0 else float(round(val))
    return out


def generate_lhs_samples(n_samples: int, seed: int) -> list[dict[str, float]]:
    """Generate stratified Latin Hypercube samples in the 10-D setup space."""
    if n_samples <= 0:
        return []
    dim = len(PARAM_NAMES)
    if HAS_SCIPY and qmc is not None:
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        unit_pts = sampler.random(n=n_samples)
    else:
        rng = np.random.default_rng(seed)
        unit_pts = np.empty((n_samples, dim), dtype=np.float64)
        for j in range(dim):
            perm = rng.permutation(n_samples)
            jitter = rng.random(n_samples)
            unit_pts[:, j] = (perm + jitter) / float(n_samples)
    return [unit_to_params(row) for row in unit_pts]


def _extract_result_json(stdout_text: str) -> dict[str, Any] | None:
    """Parse SETUP_OPT_RESULT:{...} line emitted by eval_setup_candidate.gd."""
    return extract_result_json(stdout_text, RESULT_PREFIX)


def compute_physical_penalty(metrics: dict[str, Any]) -> tuple[bool, float]:
    """Compute strict feasibility flag and physical constraint penalties."""
    ok = bool(metrics.get("ok", False))
    min_fz = float(metrics.get("min_fz_n", 0.0))
    max_roll = float(metrics.get("max_roll_deg", 99.0))
    peak_beta = float(metrics.get("peak_beta_deg", 99.0))
    hs_beta = float(metrics.get("high_speed_beta_max_deg", 99.0))
    brk_lock = float(metrics.get("brake_lockup_peak", 9.9))

    penalty = 0.0
    if not ok:
        penalty += 10000.0
    if not math.isfinite(min_fz) or min_fz < 110.0:
        val_fz = min_fz if math.isfinite(min_fz) else 0.0
        penalty += 1000.0 + 30.0 * (110.0 - val_fz)
    if not math.isfinite(max_roll) or max_roll > 1.6:
        val_roll = max_roll if math.isfinite(max_roll) else 10.0
        penalty += 1000.0 + 400.0 * (val_roll - 1.6)
    if not math.isfinite(peak_beta) or peak_beta > 4.5:
        val_beta = peak_beta if math.isfinite(peak_beta) else 30.0
        penalty += 1000.0 + 250.0 * (val_beta - 4.5)
    if not math.isfinite(hs_beta) or hs_beta > 3.5:
        val_hs = hs_beta if math.isfinite(hs_beta) else 30.0
        penalty += 1000.0 + 300.0 * (val_hs - 3.5)
    if not math.isfinite(brk_lock) or brk_lock > 0.16:
        val_lock = brk_lock if math.isfinite(brk_lock) else 1.0
        penalty += 1000.0 + 2500.0 * (val_lock - 0.16)

    feasible = (
        ok
        and math.isfinite(min_fz)
        and math.isfinite(max_roll)
        and math.isfinite(peak_beta)
        and math.isfinite(hs_beta)
        and math.isfinite(brk_lock)
        and min_fz >= 110.0
        and max_roll <= 1.6
        and peak_beta <= 4.5
        and hs_beta <= 3.5
        and brk_lock <= 0.16
    )
    return feasible, penalty


def _run_single_evaluation(job: EvalJob) -> dict[str, Any]:
    """Execute one headless Godot evaluation using a temporary params JSON file."""
    t0 = time.perf_counter()
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"setup_cand_{job.index}_",
            encoding="utf-8",
            delete=False,
        ) as tf:
            json.dump({"params": job.params}, tf)
            tmp_path = tf.name

        cmd = [
            job.godot_bin,
            "--headless",
            "--path",
            job.project_dir,
            "-s",
            EVAL_SCRIPT_RES,
            "--",
            f"--setup={tmp_path}",
            f"--map={job.map_id}",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=job.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "index": job.index,
            "label": job.label,
            "ok": False,
            "feasible": False,
            "penalty": 20000.0,
            "error": f"timeout after {job.timeout_s:.1f}s",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "params": job.params,
            "metrics": {},
        }
    except OSError as exc:
        return {
            "index": job.index,
            "label": job.label,
            "ok": False,
            "feasible": False,
            "penalty": 20000.0,
            "error": f"launch failed: {exc}",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "params": job.params,
            "metrics": {},
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    elapsed_s = round(time.perf_counter() - t0, 3)
    parsed = _extract_result_json(proc.stdout)
    if parsed is None:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return {
            "index": job.index,
            "label": job.label,
            "ok": False,
            "feasible": False,
            "penalty": 20000.0,
            "error": f"missing {RESULT_PREFIX} (rc={proc.returncode}): {tail}",
            "elapsed_s": elapsed_s,
            "params": job.params,
            "metrics": {},
        }

    feasible, penalty = compute_physical_penalty(parsed)
    metrics = {k: v for k, v in parsed.items() if k != "params"}
    metrics["feasible"] = feasible
    metrics["penalty"] = penalty

    return {
        "index": job.index,
        "label": job.label,
        "ok": bool(parsed.get("ok", False)),
        "feasible": feasible,
        "penalty": penalty,
        "elapsed_s": elapsed_s,
        "params": dict(job.params),
        "metrics": metrics,
    }


def run_batch(
    jobs: list[EvalJob],
    max_workers: int,
    total_budget: int,
) -> list[dict[str, Any]]:
    """Run a batch of Godot candidate evaluations in parallel."""
    results: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(jobs)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run_single_evaluation, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            res = future.result()
            results.append(res)
            if res["ok"]:
                m = res["metrics"]
                feas_tag = "FEAS" if res["feasible"] else f"PEN={res['penalty']:.0f}"
                print(
                    f"[eval {res['index']:2d}/{total_budget - 1:2d}] ({res['label']:10s}) "
                    f"{feas_tag:10s} {res['elapsed_s']:4.2f}s | "
                    f"skid={m['skidpad_lap_time_s']:.3f}s ({m['skidpad_lat_g']:.2f}g) "
                    f"slal_wz={m['slalom_wz_rmse_rads']:.4f} "
                    f"brk={m['trail_brake_dist_m']:.2f}m/{m['trail_brake_time_s']:.3f}s "
                    f"eff={m['efficiency_kwh_km']:.4f} "
                    f"Fz={m['min_fz_n']:.0f}N roll={m['max_roll_deg']:.2f}d beta={m['peak_beta_deg']:.2f}d",
                    flush=True,
                )
            else:
                print(
                    f"[eval {res['index']:2d}/{total_budget - 1:2d}] ({res['label']:10s}) "
                    f"FAIL ({res['elapsed_s']:.2f}s): {res.get('error', 'unknown')}",
                    flush=True,
                )
    results.sort(key=lambda r: r["index"])
    return results


def compute_metric_scales(evaluated: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Compute reference (lo, span) scales from valid evaluations for balanced weighting."""
    pool = [r for r in evaluated if r["feasible"]]
    if not pool:
        pool = [r for r in evaluated if r["ok"]]
    floors = {
        "skidpad_lap_time_s": 0.04,
        "skidpad_wz_rmse": 0.02,
        "slalom_wz_rmse_rads": 0.008,
        "trail_brake_time_s": 0.03,
        "trail_brake_dist_m": 0.25,
        "high_speed_wz_rmse": 0.02,
        "high_speed_beta_max_deg": 0.20,
        "efficiency_kwh_km": 0.004,
        "peak_beta_deg": 0.30,
    }
    scales: dict[str, tuple[float, float]] = {}
    for key, floor_span in floors.items():
        vals = [
            float(r["metrics"][key])
            for r in pool
            if key in r["metrics"] and math.isfinite(float(r["metrics"][key]))
        ]
        if vals:
            lo = min(vals)
            span = max(max(vals) - lo, floor_span)
        else:
            lo, span = 0.0, floor_span
        scales[key] = (lo, span)
    return scales


def evaluate_event_objective(
    event_key: str,
    metrics: dict[str, Any],
    scales: dict[str, tuple[float, float]],
) -> float:
    """Scalarized objective (minimize) for Skidpad, Autocross, or Endurance."""
    _, penalty = compute_physical_penalty(metrics)
    skid_t = float(metrics.get("skidpad_lap_time_s", 99.0))
    skid_wz = float(metrics.get("skidpad_wz_rmse", 9.9))
    skid_beta = float(metrics.get("skidpad_peak_beta_deg", 15.0))
    slal_wz = float(metrics.get("slalom_wz_rmse_rads", 9.9))
    trail_t = float(metrics.get("trail_brake_time_s", 9.9))
    trail_d = float(metrics.get("trail_brake_dist_m", 99.0))
    eff = float(metrics.get("efficiency_kwh_km", 9.9))
    peak_beta = float(metrics.get("peak_beta_deg", 15.0))
    hs_wz = float(metrics.get("high_speed_wz_rmse", 9.9))
    hs_beta = float(metrics.get("high_speed_beta_max_deg", 15.0))
    rear_lock = float(metrics.get("rear_lock_margin", 0.0))

    def _norm(key: str, val: float) -> float:
        lo, span = scales.get(key, (0.0, 1.0))
        return (val - lo) / span

    rear_lock_soft = max(0.0, -rear_lock) * 5.0

    if event_key == "opt_skidpad_setup":
        return skid_t + 0.25 * skid_wz + 0.02 * skid_beta + 0.03 * hs_beta + penalty

    if event_key == "opt_autocross_setup":
        score = (
            0.30 * _norm("skidpad_lap_time_s", skid_t)
            + 0.22 * _norm("slalom_wz_rmse_rads", slal_wz)
            + 0.18 * _norm("trail_brake_time_s", trail_t)
            + 0.05 * _norm("trail_brake_dist_m", trail_d)
            + 0.13 * _norm("peak_beta_deg", peak_beta)
            + 0.12 * _norm("high_speed_wz_rmse", hs_wz)
        )
        return score + 0.05 * rear_lock_soft + penalty

    if event_key == "opt_endurance_setup":
        score = (
            0.35 * _norm("efficiency_kwh_km", eff)
            + 0.22 * _norm("peak_beta_deg", peak_beta)
            + 0.18 * _norm("slalom_wz_rmse_rads", slal_wz)
            + 0.13 * _norm("skidpad_lap_time_s", skid_t)
            + 0.12 * _norm("high_speed_beta_max_deg", hs_beta)
        )
        return score + 0.05 * rear_lock_soft + penalty

    raise ValueError(f"Unknown event_key: {event_key}")


class NumpyRBFSurrogate:
    """Lightweight ridge-regularized multiquadric RBF surrogate fallback (pure NumPy)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, reg: float = 1e-4) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.y_mean = float(np.mean(y))
        y_c = np.asarray(y, dtype=np.float64) - self.y_mean
        diffs = self.X[:, None, :] - self.X[None, :, :]
        r2 = np.sum(diffs * diffs, axis=-1)
        self.eps = 0.75
        Phi = np.sqrt(r2 + self.eps * self.eps)
        n = len(self.X)
        A = Phi + reg * np.eye(n)
        self.weights = np.linalg.solve(A, y_c)

    def __call__(self, Xq: np.ndarray) -> np.ndarray:
        Xq = np.atleast_2d(np.asarray(Xq, dtype=np.float64))
        diffs = Xq[:, None, :] - self.X[None, :, :]
        r2 = np.sum(diffs * diffs, axis=-1)
        Phi = np.sqrt(r2 + self.eps * self.eps)
        return self.y_mean + Phi @ self.weights


def fit_surrogates(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit one RBF surrogate per metric on all valid simulation evaluations."""
    valid = [r for r in evaluated if r["ok"]]
    if len(valid) < 2:
        return {}

    X = np.vstack([params_to_unit(r["params"]) for r in valid])
    models: dict[str, Any] = {}
    dim = X.shape[1]

    for mkey in SURROGATE_METRICS:
        y = np.array([float(r["metrics"].get(mkey, 0.0)) for r in valid], dtype=np.float64)
        if not np.all(np.isfinite(y)):
            continue
        if HAS_SCIPY and RBFInterpolator is not None and len(valid) >= dim + 2:
            try:
                models[mkey] = RBFInterpolator(
                    X, y, kernel="thin_plate_spline", degree=1, smoothing=1e-4
                )
                continue
            except Exception:
                pass
        models[mkey] = NumpyRBFSurrogate(X, y, reg=1e-3)
    return models


def _predict_metrics_batch(models: dict[str, Any], Xq: np.ndarray) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for mkey, mdl in models.items():
        preds[mkey] = np.asarray(mdl(Xq), dtype=np.float64)
    return preds


def _eval_surrogate_objective_vec(
    event_key: str,
    preds: dict[str, np.ndarray],
    scales: dict[str, tuple[float, float]],
    u_mat: np.ndarray | None = None,
) -> np.ndarray:
    n = len(next(iter(preds.values())))
    min_fz = preds.get("min_fz_n", np.full(n, 200.0))
    max_roll = preds.get("max_roll_deg", np.full(n, 1.0))
    peak_beta = preds.get("peak_beta_deg", np.full(n, 3.0))
    hs_beta = preds.get("high_speed_beta_max_deg", np.full(n, 1.5))
    brk_lock = preds.get("brake_lockup_peak", np.full(n, 0.05))

    penalty = np.zeros(n, dtype=np.float64)
    mask_fz = min_fz < 110.0
    penalty[mask_fz] += 1000.0 + 30.0 * (110.0 - min_fz[mask_fz])
    mask_roll = max_roll > 1.6
    penalty[mask_roll] += 1000.0 + 400.0 * (max_roll[mask_roll] - 1.6)
    mask_beta = peak_beta > 4.5
    penalty[mask_beta] += 1000.0 + 250.0 * (peak_beta[mask_beta] - 4.5)
    mask_hs = hs_beta > 3.5
    penalty[mask_hs] += 1000.0 + 300.0 * (hs_beta[mask_hs] - 3.5)
    mask_lock = brk_lock > 0.16
    penalty[mask_lock] += 1000.0 + 2500.0 * (brk_lock[mask_lock] - 0.16)

    # Gentle interior regularization to avoid L-BFGS-B pegging on flat RBF plateaus
    interior_pen = np.zeros(n, dtype=np.float64)
    if u_mat is not None:
        edge_viol = np.maximum(0.0, 0.06 - u_mat) + np.maximum(0.0, u_mat - 0.94)
        interior_pen = 0.25 * np.sum(edge_viol * edge_viol, axis=1) + 0.002 * np.sum(
            (u_mat - 0.5) ** 2, axis=1
        )

    def _norm(key: str, arr: np.ndarray) -> np.ndarray:
        lo, span = scales.get(key, (0.0, 1.0))
        return (arr - lo) / span

    rear_lock = preds.get("rear_lock_margin", np.zeros(n))
    rear_lock_soft = np.maximum(0.0, -rear_lock) * 5.0

    skid_t = preds["skidpad_lap_time_s"]
    skid_wz = preds["skidpad_wz_rmse"]
    skid_beta = preds["skidpad_peak_beta_deg"]
    slal_wz = preds["slalom_wz_rmse_rads"]
    trail_t = preds["trail_brake_time_s"]
    trail_d = preds["trail_brake_dist_m"]
    hs_wz = preds.get("high_speed_wz_rmse", np.zeros(n))
    eff = preds["efficiency_kwh_km"]

    if event_key == "opt_skidpad_setup":
        return skid_t + 0.25 * skid_wz + 0.02 * skid_beta + 0.03 * hs_beta + interior_pen + penalty
    if event_key == "opt_autocross_setup":
        return (
            0.30 * _norm("skidpad_lap_time_s", skid_t)
            + 0.22 * _norm("slalom_wz_rmse_rads", slal_wz)
            + 0.18 * _norm("trail_brake_time_s", trail_t)
            + 0.05 * _norm("trail_brake_dist_m", trail_d)
            + 0.13 * _norm("peak_beta_deg", peak_beta)
            + 0.12 * _norm("high_speed_wz_rmse", hs_wz)
            + 0.05 * rear_lock_soft
            + interior_pen
            + penalty
        )
    if event_key == "opt_endurance_setup":
        return (
            0.35 * _norm("efficiency_kwh_km", eff)
            + 0.22 * _norm("peak_beta_deg", peak_beta)
            + 0.18 * _norm("slalom_wz_rmse_rads", slal_wz)
            + 0.13 * _norm("skidpad_lap_time_s", skid_t)
            + 0.12 * _norm("high_speed_beta_max_deg", hs_beta)
            + 0.05 * rear_lock_soft
            + interior_pen
            + penalty
        )
    raise ValueError(f"Unknown event_key: {event_key}")


def propose_surrogate_candidates(
    evaluated: list[dict[str, Any]],
    models: dict[str, Any],
    scales: dict[str, tuple[float, float]],
    event_keys: tuple[str, ...],
    rng: np.random.Generator,
) -> list[tuple[str, dict[str, float]]]:
    """Propose one polished surrogate-optimal candidate per requested event."""
    if not models:
        return []

    existing_u = np.vstack([params_to_unit(r["params"]) for r in evaluated])
    dim = len(PARAM_NAMES)

    # Dense cloud: uniform global samples + local Gaussian perturbations around feasible elites
    n_global = 4096
    X_cloud = rng.random((n_global, dim))
    valid_pool = [r for r in evaluated if r["ok"]]
    if valid_pool:
        elite_pts = [params_to_unit(r["params"]) for r in valid_pool[: min(6, len(valid_pool))]]
        local_chunks = [
            np.clip(ep[None, :] + rng.normal(0.0, 0.12, size=(128, dim)), 0.0, 1.0)
            for ep in elite_pts
        ]
        X_cloud = np.vstack([X_cloud, *local_chunks])

    preds_cloud = _predict_metrics_batch(models, X_cloud)
    proposals: list[tuple[str, dict[str, float]]] = []
    proposed_u: list[np.ndarray] = []

    for ev in event_keys:
        scores = _eval_surrogate_objective_vec(ev, preds_cloud, scales, X_cloud)
        top_idxs = np.argsort(scores)[:5]

        best_u = X_cloud[int(top_idxs[0])].copy()
        best_val = float(scores[int(top_idxs[0])])

        def _obj_single(u_vec: np.ndarray) -> float:
            u_c = np.clip(u_vec, 0.0, 1.0)[None, :]
            p_one = _predict_metrics_batch(models, u_c)
            return float(_eval_surrogate_objective_vec(ev, p_one, scales, u_c)[0])

        if HAS_SCIPY and scipy_minimize is not None:
            for idx in top_idxs[:3]:
                x0 = X_cloud[int(idx)]
                res_opt = scipy_minimize(
                    _obj_single,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(0.0, 1.0)] * dim,
                    options={"maxiter": 60},
                )
                if float(res_opt.fun) < best_val:
                    best_val = float(res_opt.fun)
                    best_u = np.clip(np.asarray(res_opt.x, dtype=np.float64), 0.0, 1.0)

        # Ensure diversity from already-evaluated and already-proposed points
        all_seen = np.vstack([existing_u, *proposed_u]) if proposed_u else existing_u
        min_d = float(np.min(np.linalg.norm(all_seen - best_u[None, :], axis=1)))
        if min_d < 0.025:
            # Pick best diverse cloud candidate at least 0.04 away in normalized space
            dists = np.min(np.linalg.norm(all_seen[:, None, :] - X_cloud[None, :, :], axis=2), axis=0)
            diverse_order = np.argsort(scores + np.where(dists < 0.04, 1e6, 0.0))
            best_u = X_cloud[int(diverse_order[0])].copy()

        proposed_u.append(best_u)
        proposals.append((ev.replace("opt_", "").replace("_setup", ""), unit_to_params(best_u)))

    return proposals


def select_best_for_events(evaluated: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select the best genuinely evaluated Godot candidate for each of the 3 events."""
    scales = compute_metric_scales(evaluated)
    pool = [r for r in evaluated if r["feasible"]]
    if not pool:
        pool = [r for r in evaluated if r["ok"]]
    if not pool:
        pool = evaluated

    winners: dict[str, dict[str, Any]] = {}
    for ev in EVENT_SPECS:
        best_entry = min(
            pool,
            key=lambda r: evaluate_event_objective(ev, r["metrics"], scales),
        )
        score = evaluate_event_objective(ev, best_entry["metrics"], scales)
        winners[ev] = {
            "event_key": ev,
            "score": round(score, 6),
            "sample_index": best_entry["index"],
            "label": best_entry["label"],
            "params": best_entry["params"],
            "metrics": best_entry["metrics"],
        }
    return winners


def export_presets(
    output_dir: Path,
    winners: dict[str, dict[str, Any]],
) -> list[Path]:
    """Write opt_skidpad_setup.json, opt_autocross_setup.json, opt_endurance_setup.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ev_key, spec in EVENT_SPECS.items():
        win = winners.get(ev_key)
        if not win:
            continue
        payload = {
            "name": spec["name"],
            "description": spec["description"],
            "author": "optimize_vehicle_setup.py",
            "params": win["params"],
            "metrics": win["metrics"],
        }
        out_path = output_dir / f"{ev_key}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(out_path)
    return written


def print_comparison_table(
    baseline: dict[str, Any],
    winners: dict[str, dict[str, Any]],
    evaluated: list[dict[str, Any]],
    wall_time_s: float,
    exported_paths: list[Path],
) -> None:
    """Print clear Baseline vs Optimal comparison tables in stdout."""
    n_ok = sum(1 for r in evaluated if r["ok"])
    n_feas = sum(1 for r in evaluated if r["feasible"])

    print("\n" + "=" * 108)
    print("  MIRENA DYNAMICS (TeR27) — MULTI-OBJECTIVE VEHICLE SETUP OPTIMIZATION (BASELINE vs OPTIMAL)")
    print("=" * 108)
    print(
        f"Evaluations : {len(evaluated)} total | {n_ok} ok | {n_feas} feasible "
        f"(min_fz>=110N, max_roll<=1.6deg, peak_beta<=4.5deg, hs_beta<=3.5deg, brk_lock<=0.16)"
    )
    print(
        f"Surrogate   : {'SciPy RBFInterpolator (TPS) + L-BFGS-B' if HAS_SCIPY else 'NumPy Multiquadric RBF'} "
        f"| Wall time: {wall_time_s:.2f} s\n"
    )

    bm = baseline.get("metrics", {})
    print("[METRICS COMPARISON: BASELINE vs OPTIMAL PRESETS]")
    print(
        "  Setup                 | #   | Skid(s) | LatG  | SlalRMSE | Brk(m) | HSBeta | BrkLck | kWh/km | MinFz | Roll  | Beta  | Feas"
    )
    print("  " + "-" * 114)

    rows: list[tuple[str, dict[str, Any]]] = [("Baseline", baseline)]
    for ev_key in ("opt_skidpad_setup", "opt_autocross_setup", "opt_endurance_setup"):
        if ev_key in winners:
            rows.append((EVENT_SPECS[ev_key]["name"], winners[ev_key]))

    for name, entry in rows:
        m = entry.get("metrics", {})
        idx = entry.get("sample_index", entry.get("index", 0))
        feas_str = "YES" if m.get("feasible", False) else "NO"
        print(
            f"  {name:21s} | {idx:3d} | "
            f"{float(m.get('skidpad_lap_time_s', float('nan'))):7.3f} | "
            f"{float(m.get('skidpad_lat_g', float('nan'))):5.3f} | "
            f"{float(m.get('slalom_wz_rmse_rads', float('nan'))):8.5f} | "
            f"{float(m.get('trail_brake_dist_m', float('nan'))):6.2f} | "
            f"{float(m.get('high_speed_beta_max_deg', float('nan'))):5.2f}d | "
            f"{float(m.get('brake_lockup_peak', float('nan'))):6.4f} | "
            f"{float(m.get('efficiency_kwh_km', float('nan'))):6.4f} | "
            f"{float(m.get('min_fz_n', float('nan'))):5.0f} | "
            f"{float(m.get('max_roll_deg', float('nan'))):4.2f}d | "
            f"{float(m.get('peak_beta_deg', float('nan'))):4.2f}d | {feas_str}"
        )

    print("\n[PARAMETER COMPARISON: BASELINE vs OPTIMAL PRESETS]")
    print(
        "  Setup                 | kF(N/m) | kR(N/m) | arbF  | arbR  | camF  | camR  | toeF   | toeR   | brkBias | aeroF"
    )
    print("  " + "-" * 104)
    for name, entry in rows:
        p = entry.get("params", {})
        print(
            f"  {name:21s} | "
            f"{float(p.get('spring_stiffness_front', 0)):7.0f} | "
            f"{float(p.get('spring_stiffness_rear', 0)):7.0f} | "
            f"{float(p.get('arb_stiffness_front', 0)):5.0f} | "
            f"{float(p.get('arb_stiffness_rear', 0)):5.0f} | "
            f"{float(p.get('static_camber_front_deg', 0)):5.2f} | "
            f"{float(p.get('static_camber_rear_deg', 0)):5.2f} | "
            f"{float(p.get('toe_front_deg', 0)):6.3f} | "
            f"{float(p.get('toe_rear_deg', 0)):6.3f} | "
            f"{float(p.get('brake_bias_front', 0)):7.4f} | "
            f"{float(p.get('aero_balance_front', 0)):5.3f}"
        )

    if bm:
        print("\n[DELTA vs BASELINE]")
        for ev_key in ("opt_skidpad_setup", "opt_autocross_setup", "opt_endurance_setup"):
            win = winners.get(ev_key)
            if not win:
                continue
            wm = win["metrics"]
            d_skid = float(wm.get("skidpad_lap_time_s", 0.0)) - float(bm.get("skidpad_lap_time_s", 0.0))
            d_slal = float(wm.get("slalom_wz_rmse_rads", 0.0)) - float(bm.get("slalom_wz_rmse_rads", 0.0))
            d_brk = float(wm.get("trail_brake_dist_m", 0.0)) - float(bm.get("trail_brake_dist_m", 0.0))
            d_eff = float(wm.get("efficiency_kwh_km", 0.0)) - float(bm.get("efficiency_kwh_km", 0.0))
            print(
                f"  {EVENT_SPECS[ev_key]['name']:21s}: "
                f"skidpad {d_skid:+.3f} s | slalom_wzRMSE {d_slal:+.5f} rad/s | "
                f"trail_dist {d_brk:+.2f} m | eff {d_eff:+.4f} kWh/km"
            )

    if exported_paths:
        print("\n[EXPORTED SETUP PRESETS]")
        for p in exported_paths:
            print(f"  -> {p}")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-objective mechanical & brake setup optimizer via headless Godot."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=24,
        help="Total simulation evaluation budget across LHS + surrogate refinement (default: 24).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 2)),
        help="Parallel Godot worker processes (default: min(4, cpu_count)).",
    )
    parser.add_argument(
        "--godot",
        "--godot-bin",
        dest="godot",
        type=str,
        default="",
        help="Optional explicit path to godot binary.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for opt_*_setup.json presets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LHS & surrogate search (default: 42).",
    )
    parser.add_argument(
        "--map-id",
        type=int,
        default=2,
        help="ECU TV V3 map ID used during closed-loop evaluation (default: 2).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-evaluation Godot subprocess timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Optional JSON report path containing all evaluated candidates and winners.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    budget = max(1, int(args.budget))
    godot_bin = resolve_godot_bin(args.godot or None)
    project_dir = str(DEFAULT_PROJECT_DIR)
    rng = np.random.default_rng(args.seed)

    print(
        f"[optimize_vehicle_setup] budget={budget}, workers={args.workers}, "
        f"scipy={HAS_SCIPY}, godot={godot_bin}",
        flush=True,
    )

    t_wall0 = time.perf_counter()
    baseline_params = load_baseline_from_config(DEFAULT_PROJECT_DIR)

    # Split budget across Phase 1 (Baseline + LHS) and Phase 2/3 (Surrogate refinement rounds)
    if budget >= 12:
        n_surr_total = 6
    elif budget >= 6:
        n_surr_total = 3
    else:
        n_surr_total = 0

    n_phase1 = budget - n_surr_total
    n_lhs = max(0, n_phase1 - 1)
    phase1_params: list[tuple[str, dict[str, float]]] = [("baseline", baseline_params)]
    for cand in generate_lhs_samples(n_lhs, args.seed):
        phase1_params.append(("lhs", cand))

    print(
        f"\n--- Phase 1: Baseline + {n_lhs} LHS Exploration Evaluations ---",
        flush=True,
    )
    jobs_p1 = [
        EvalJob(
            index=i,
            label=lbl,
            params=p,
            project_dir=project_dir,
            godot_bin=godot_bin,
            map_id=args.map_id,
            timeout_s=args.timeout,
        )
        for i, (lbl, p) in enumerate(phase1_params)
    ]
    evaluated = run_batch(jobs_p1, args.workers, budget)

    # Phase 2 & 3: Surrogate RBF fitting + iterative real Godot validation
    next_idx = len(evaluated)
    remaining = budget - next_idx
    event_keys = ("opt_skidpad_setup", "opt_autocross_setup", "opt_endurance_setup")

    round_num = 1
    while remaining > 0:
        models = fit_surrogates(evaluated)
        if not models:
            # Fallback to LHS if fewer than 2 valid points exist
            extra = generate_lhs_samples(remaining, args.seed + round_num * 101)
            jobs_fb = [
                EvalJob(
                    index=next_idx + i,
                    label="lhs_fb",
                    params=p,
                    project_dir=project_dir,
                    godot_bin=godot_bin,
                    map_id=args.map_id,
                    timeout_s=args.timeout,
                )
                for i, p in enumerate(extra)
            ]
            evaluated.extend(run_batch(jobs_fb, args.workers, budget))
            break

        scales = compute_metric_scales(evaluated)
        batch_events = event_keys[: min(3, remaining)]
        print(
            f"\n--- Phase 2/3 Round {round_num}: Surrogate RBF Optimization -> {len(batch_events)} Real Godot Validations ---",
            flush=True,
        )
        proposals = propose_surrogate_candidates(evaluated, models, scales, batch_events, rng)
        jobs_p2 = [
            EvalJob(
                index=next_idx + i,
                label=f"surr_{tag}",
                params=p,
                project_dir=project_dir,
                godot_bin=godot_bin,
                map_id=args.map_id,
                timeout_s=args.timeout,
            )
            for i, (tag, p) in enumerate(proposals)
        ]
        evaluated.extend(run_batch(jobs_p2, args.workers, budget))
        next_idx = len(evaluated)
        remaining = budget - next_idx
        round_num += 1

    evaluated.sort(key=lambda r: r["index"])
    wall_time_s = time.perf_counter() - t_wall0

    baseline_entry = evaluated[0]
    winners = select_best_for_events(evaluated)
    exported_paths = export_presets(Path(args.output_dir), winners)

    print_comparison_table(baseline_entry, winners, evaluated, wall_time_s, exported_paths)

    if args.report.strip():
        report_path = Path(args.report.strip())
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "budget": budget,
            "workers": args.workers,
            "wall_time_s": round(wall_time_s, 3),
            "scipy_available": HAS_SCIPY,
            "baseline": baseline_entry,
            "winners": winners,
            "all_evaluations": evaluated,
        }
        report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved optimization report to {report_path}")

    any_feasible = any(r["feasible"] for r in evaluated)
    return 0 if any_feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
