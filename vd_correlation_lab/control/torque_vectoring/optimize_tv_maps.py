#!/usr/bin/env python3
"""
Deterministic Offline TV Map Optimizer (seed=42) over Full14DofModel.gd (3D kinematics):
  1. SKIDPAD map optimization (Skidpad R=9.125 m lap time + yaw overshoot & beta guardrails)
  2. FSG 2023 AUTOCROSS map optimization (FSG 2023 14-DOF lap time + DLC 20 / Bode / HV / TF guardrails)
  3. FSG 2023 EFFICIENCY Pareto front with speed-dependent P_max(vx) minimizing J = E_22km * T_22km^2

Outputs:
  - vd_correlation_lab/results/lts/tv_map_optimization_summary.json
  - vd_correlation_lab/results/lts/07_tv_map_optimization_pareto.png (300 DPI)
Updates tv_4wd_config.json only for modes whose validated winner improves the baseline and passes all guardrails.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

LAB_DIR = Path(__file__).resolve().parents[2]
LTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parent
OUT_DIR = LAB_DIR / "results" / "lts"
TV4_CFG_PATH = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Config" / "vehicles" / "ter27_4wd" / "tv_4wd_config.json"
GUARDRAILS_GD = LTS_DIR / "godot_guardrails_probe.gd"
FSG_2023_TRACK_PATH = REPO_ROOT / "data" / "tracks" / "fsg_2023.json"
SIM_DIR = REPO_ROOT / "src" / "mirena_sim"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from vd_correlation_lab.analysis.visualization.style import C_ACCENT, C_HORIZON, C_MUTED, C_OPEN, C_TER27, apply_style, save_fig  # noqa: E402
from vd_correlation_lab.optimization.lap_time_sim import (  # noqa: E402
    Track,
    extract_or_load_ggv,
    optimize_racing_line,
    project_endurance_22km,
    run_transient_lap,
    search_fastest_valid_lap,
    solve_qss,
)
from vd_correlation_lab.optimization.lap_time_sim.ggv_extractor import GGVMap  # noqa: E402
from vd_correlation_lab.optimization.lap_time_sim.tracks.build_fsg_2023 import main as build_fsg_main  # noqa: E402

SEED = 42


def _emit_progress(pf_path: Path | None, payload: Dict[str, Any]) -> None:
    if pf_path is None:
        return
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    with pf_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
        fh.flush()


def _analyze_bode_csv(df: pd.DataFrame) -> float:
    t = df["t_s"].values
    u = np.radians(df["steer_deg"].values)
    r = df["wz"].values
    f_grid = np.linspace(0.25, 3.25, 31)
    mag_r = []
    for f0 in f_grid:
        t_center = 8.0 * (f0 - 0.2) / 3.3
        mask = (t >= max(0.2, t_center - 0.9)) & (t <= min(7.8, t_center + 0.9))
        if np.sum(mask) < 20:
            mag_r.append(0.0)
            continue
        u_w = u[mask] * np.hanning(int(np.sum(mask)))
        r_w = r[mask] * np.hanning(int(np.sum(mask)))
        U = np.fft.rfft(u_w)
        R = np.fft.rfft(r_w)
        idx = int(np.argmax(np.abs(U)))
        mag_r.append(float(np.abs(R[idx] / (U[idx] + 1e-9))))
    mag_arr = np.asarray(mag_r, dtype=float)
    dc = float(mag_arr[0]) if mag_arr[0] > 0.0 else 1.0
    return float(20.0 * np.log10(np.max(mag_arr) / max(dc, 1e-6)))


def _run_guardrails(mode_overrides: Dict[str, Any], tasks: str = "skidpad,dlc20,bode") -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mirena_tv_opt_") as tmp:
        tmp_dir = Path(tmp)
        cfg_p = tmp_dir / "cfg.json"
        cfg_p.write_text(json.dumps({"mode_overrides": mode_overrides}), encoding="utf-8")
        cmd = [
            "godot", "--headless", "--path", str(SIM_DIR), "-s", str(GUARDRAILS_GD),
            f"--cfg={cfg_p}", f"--out_dir={tmp_dir}", f"--tasks={tasks}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out: Dict[str, Any] = {}
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        if "skidpad" in task_list:
            df = pd.read_csv(tmp_dir / "skidpad.csv")
            ss = df[(df["t_s"] >= 2.0) & (df["t_s"] <= 6.0)]
            v_ss = float(ss["vx"].mean())
            t_lap = (2.0 * np.pi * 9.125) / max(v_ss, 1.0)
            tr = df[(df["t_s"] >= 6.5) & (df["t_s"] <= 8.5)]
            ov = max(0.0, float(tr["wz"].abs().max() - abs(tr["wz"].iloc[-1])))
            max_beta = float(df["beta_deg"].abs().max())
            p_min = float(df["p_dc_kw"].min())
            p_max = float(df["p_dc_kw"].max())
            out["skidpad"] = {
                "t_lap_s": round(t_lap, 4),
                "vx_ss_ms": round(v_ss, 3),
                "trans_wz_overshoot": round(ov, 4),
                "max_abs_beta_deg": round(max_beta, 3),
                "p_dc_min_kw": round(p_min, 2),
                "p_dc_max_kw": round(p_max, 2),
            }
        if "dlc20" in task_list:
            df_d = pd.read_csv(tmp_dir / "dlc20.csv")
            out["dlc20"] = {
                "max_abs_beta_deg": round(float(df_d["beta_deg"].abs().max()), 3),
                "p_dc_min_kw": round(float(df_d["p_dc_kw"].min()), 2),
                "p_dc_max_kw": round(float(df_d["p_dc_kw"].max()), 2),
            }
        if "bode" in task_list:
            df_b = pd.read_csv(tmp_dir / "bode.csv")
            out["bode"] = {
                "res_peak_db": round(_analyze_bode_csv(df_b), 3),
                "p_dc_min_kw": round(float(df_b["p_dc_kw"].min()), 2),
                "p_dc_max_kw": round(float(df_b["p_dc_kw"].max()), 2),
            }
        return out


def _scale_ggv_for_power_curve(ggv_base: GGVMap, p_lo_kw: float, p_hi_kw: float, v_lo: float, v_hi: float) -> GGVMap:
    ax_new = ggv_base.ax_max.copy()
    eta_drv = 0.90
    for iv, v in enumerate(ggv_base.speeds):
        u = float(np.clip((v - v_lo) / max(v_hi - v_lo, 0.5), 0.0, 1.0))
        p_kw = p_lo_kw + (p_hi_kw - p_lo_kw) * u
        p_w = min(80000.0, max(5000.0, p_kw * 1000.0))
        ax_cap = (p_w * eta_drv) / (ggv_base.mass_kg * max(float(v), 2.0))
        ax_new[iv, :] = np.minimum(ax_new[iv, :], ax_cap)
    return replace(ggv_base, ax_max=ax_new)


def _lhs_samples(bounds: List[Tuple[float, float]], n_samples: int, seed: int) -> np.ndarray:
    sampler = LatinHypercube(d=len(bounds), seed=seed)
    u = sampler.random(n=n_samples)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + u * (hi - lo)


def optimize_skidpad(
    base_cfg: Dict[str, Any],
    n_lhs: int = 10,
    n_refine: int = 4,
    pf_path: Path | None = None,
) -> Dict[str, Any]:
    print("\n[OPT 1/3] Optimizing SKIDPAD map (seed=42)...")
    keys = ["k_us_low_deg_g", "k_us_deg_g", "dynamic_rear_bias_gain", "k_ff_low_mult", "beta_soft_deg"]
    bounds = [(0.05, 0.30), (0.15, 0.45), (0.25, 0.70), (1.00, 1.50), (2.20, 3.80)]
    total_trials = 1 + n_lhs + n_refine

    def eval_cand(params: Dict[str, float], label: str) -> Dict[str, Any]:
        g = _run_guardrails({"skidpad": params}, tasks="skidpad")["skidpad"]
        ok = (
            g["max_abs_beta_deg"] <= 4.0
            and g["trans_wz_overshoot"] <= 0.12
            and g["p_dc_min_kw"] >= -25.05
            and g["p_dc_max_kw"] <= 80.05
        )
        pen = 0.0
        if g["max_abs_beta_deg"] > 4.0:
            pen += 2.0 * (g["max_abs_beta_deg"] - 4.0)
        if g["trans_wz_overshoot"] > 0.12:
            pen += 3.0 * (g["trans_wz_overshoot"] - 0.12)
        return {
            "label": label,
            "params": {k: round(float(v), 4) for k, v in params.items()},
            "metrics": g,
            "feasible": ok,
            "score": g["t_lap_s"] + pen,
        }

    base_p = {k: float(base_cfg["modes"]["skidpad"][k]) for k in keys}
    b0 = eval_cand(base_p, "baseline")
    history = [b0]
    best = b0
    _emit_progress(pf_path, {
        "stage": "skidpad", "trial": len(history), "total_trials": total_trials,
        "lap_s": round(float(b0["metrics"]["t_lap_s"]), 4),
        "max_beta_deg": round(float(b0["metrics"]["max_abs_beta_deg"]), 3),
        "feasible": b0["feasible"], "is_best": True, "params": b0["params"],
    })
    for idx, vec in enumerate(_lhs_samples(bounds, n_lhs, SEED)):
        p = dict(zip(keys, vec))
        if p["k_us_deg_g"] < p["k_us_low_deg_g"]:
            p["k_us_deg_g"] = p["k_us_low_deg_g"]
        cand = eval_cand(p, f"lhs_{idx}")
        history.append(cand)
        is_new_best = cand["feasible"] and (not best["feasible"] or cand["score"] < best["score"])
        if is_new_best:
            best = cand
        _emit_progress(pf_path, {
            "stage": "skidpad", "trial": len(history), "total_trials": total_trials,
            "lap_s": round(float(cand["metrics"]["t_lap_s"]), 4),
            "max_beta_deg": round(float(cand["metrics"]["max_abs_beta_deg"]), 3),
            "feasible": cand["feasible"], "is_best": is_new_best, "params": cand["params"],
        })

    # Local refinement around best
    rng = np.random.default_rng(SEED)
    for r_i in range(n_refine):
        p_loc = {}
        for (k, (lo, hi)) in zip(keys, bounds):
            span = 0.12 * (hi - lo)
            p_loc[k] = float(np.clip(best["params"][k] + rng.uniform(-span, span), lo, hi))
        if p_loc["k_us_deg_g"] < p_loc["k_us_low_deg_g"]:
            p_loc["k_us_deg_g"] = p_loc["k_us_low_deg_g"]
        cand = eval_cand(p_loc, f"refine_{r_i}")
        history.append(cand)
        is_new_best = cand["feasible"] and (not best["feasible"] or cand["score"] < best["score"])
        if is_new_best:
            best = cand
        _emit_progress(pf_path, {
            "stage": "skidpad", "trial": len(history), "total_trials": total_trials,
            "lap_s": round(float(cand["metrics"]["t_lap_s"]), 4),
            "max_beta_deg": round(float(cand["metrics"]["max_abs_beta_deg"]), 3),
            "feasible": cand["feasible"], "is_best": is_new_best, "params": cand["params"],
        })

    print(f"  [SKIDPAD] Baseline={history[0]['metrics']['t_lap_s']:.3f}s | Best={best['metrics']['t_lap_s']:.3f}s (feasible={best['feasible']}) params={best['params']}")
    return {"baseline": history[0], "best": best, "history": history}


def optimize_autocross(
    base_cfg: Dict[str, Any],
    rl_fsg,
    ggv_auto: GGVMap,
    n_lhs: int = 8,
    n_refine: int = 2,
    pf_path: Path | None = None,
) -> Dict[str, Any]:
    print("\n[OPT 2/3] Optimizing FSG 2023 AUTOCROSS map (seed=42)...")
    keys = [
        "k_us_low_deg_g", "k_us_vx_low_ms", "k_us_vx_high_ms",
        "dynamic_rear_bias_gain", "k_ff_low_mult", "trail_brake_pro_turn_gain",
        "beta_soft_deg", "lift_off_regen_frac",
    ]
    bounds = [
        (0.08, 0.30), (9.5, 13.5), (18.0, 22.0),
        (0.35, 0.75), (1.15, 1.65), (0.60, 1.30),
        (2.80, 4.20), (0.00, 0.14),
    ]
    qss_base = solve_qss(rl_fsg.racing_track, ggv_auto)

    def eval_cand(params: Dict[str, float], label: str, full_alpha_search: bool = False) -> Dict[str, Any]:
        mo = {"autocross": params}
        if full_alpha_search:
            res, _ = search_fastest_valid_lap(
                label="tv4_auto", racing_line=rl_fsg, qss=qss_base,
                ecu_driving_mode=2, tv_4wd_mode=0, n_laps=1,
                use_3d_kinematics=True, mode_overrides=mo,
            )
        else:
            res = run_transient_lap(
                label="tv4_auto", racing_line=rl_fsg, qss=qss_base,
                ecu_driving_mode=2, tv_4wd_mode=0, alpha_drv=1.0, n_laps=1,
                use_3d_kinematics=True, mode_overrides=mo,
            )
        gr = _run_guardrails(mo, tasks="dlc20,bode")
        dlc_b = gr["dlc20"]["max_abs_beta_deg"]
        bode_pk = gr["bode"]["res_peak_db"]
        p_min = min(float(res.summary.get("p_chg_min_kw", 0.0)), gr["dlc20"]["p_dc_min_kw"], gr["bode"]["p_dc_min_kw"])
        p_max = max(float(res.summary.get("p_dc_max_kw", 0.0)), gr["dlc20"]["p_dc_max_kw"], gr["bode"]["p_dc_max_kw"])
        beta_lap = float(res.summary.get("max_abs_beta_deg", 99.0))
        tf_kj = float(res.summary.get("torque_fight_kj", 99.0))
        feasible = (
            bool(res.valid)
            and beta_lap <= 5.5
            and dlc_b < 4.2
            and bode_pk < 2.5
            and p_min >= -25.05
            and p_max <= 80.05
            and tf_kj < 5.0
        )
        return {
            "label": label,
            "params": {k: round(float(v), 4) for k, v in params.items()},
            "metrics": {
                "tr_lap_s": round(float(res.lap_time_s), 4),
                "valid": bool(res.valid),
                "max_abs_beta_deg": round(beta_lap, 3),
                "dlc20_max_beta_deg": dlc_b,
                "bode_res_peak_db": bode_pk,
                "e_net_kj": round(float(res.summary.get("e_net_kj", 0.0)), 2),
                "e_regen_kj": round(float(res.summary.get("e_regen_kj", 0.0)), 2),
                "torque_fight_kj": round(tf_kj, 3),
                "p_dc_min_kw": round(p_min, 2),
                "p_dc_max_kw": round(p_max, 2),
            },
            "feasible": feasible,
        }

    base_p = {k: float(base_cfg["modes"]["autocross"][k]) for k in keys}
    total_trials = 1 + n_lhs + n_refine
    b0 = eval_cand(base_p, "baseline", full_alpha_search=True)
    history = [b0]
    best = b0
    _emit_progress(pf_path, {
        "stage": "autocross", "trial": len(history), "total_trials": total_trials,
        "lap_s": round(float(b0["metrics"]["tr_lap_s"]), 4),
        "max_beta_deg": round(float(b0["metrics"]["max_abs_beta_deg"]), 3),
        "feasible": b0["feasible"], "is_best": True, "params": b0["params"],
    })
    for idx, vec in enumerate(_lhs_samples(bounds, n_lhs, SEED + 100)):
        p = dict(zip(keys, vec))
        cand = eval_cand(p, f"lhs_{idx}", full_alpha_search=False)
        history.append(cand)
        is_new_best = cand["feasible"] and (not best["feasible"] or cand["metrics"]["tr_lap_s"] < best["metrics"]["tr_lap_s"])
        if is_new_best:
            best = cand
        _emit_progress(pf_path, {
            "stage": "autocross", "trial": len(history), "total_trials": total_trials,
            "lap_s": round(float(cand["metrics"]["tr_lap_s"]), 4),
            "max_beta_deg": round(float(cand["metrics"]["max_abs_beta_deg"]), 3),
            "feasible": cand["feasible"], "is_best": is_new_best, "params": cand["params"],
        })

    # Verify top-best with full alpha search + local refinements
    rng = np.random.default_rng(SEED + 101)
    for r_i in range(n_refine):
        p_loc = {}
        for (k, (lo, hi)) in zip(keys, bounds):
            span = 0.12 * (hi - lo)
            p_loc[k] = float(np.clip(best["params"][k] + rng.uniform(-span, span), lo, hi))
        cand = eval_cand(p_loc, f"refine_{r_i}", full_alpha_search=True)
        history.append(cand)
        is_new_best = cand["feasible"] and (not best["feasible"] or cand["metrics"]["tr_lap_s"] < best["metrics"]["tr_lap_s"])
        if is_new_best:
            best = cand
        _emit_progress(pf_path, {
            "stage": "autocross", "trial": len(history), "total_trials": total_trials,
            "lap_s": round(float(cand["metrics"]["tr_lap_s"]), 4),
            "max_beta_deg": round(float(cand["metrics"]["max_abs_beta_deg"]), 3),
            "feasible": cand["feasible"], "is_best": is_new_best, "params": cand["params"],
        })

    if best["label"].startswith("lhs"):
        best = eval_cand(best["params"], best["label"] + "_verified", full_alpha_search=True)
        history.append(best)

    print(f"  [AUTOCROSS] Baseline={history[0]['metrics']['tr_lap_s']:.3f}s | Best={best['metrics']['tr_lap_s']:.3f}s (feasible={best['feasible']}) DLC={best['metrics']['dlc20_max_beta_deg']:.2f}deg Bode={best['metrics']['bode_res_peak_db']:.2f}dB")
    return {"baseline": history[0], "best": best, "history": history}


def optimize_efficiency_pareto(
    base_cfg: Dict[str, Any],
    rl_fsg,
    ggv_auto: GGVMap,
    n_lhs: int = 12,
    n_refine: int = 3,
    pf_path: Path | None = None,
) -> Dict[str, Any]:
    print("\n[OPT 3/3] Optimizing FSG 2023 EFFICIENCY Pareto front P_max(vx) (seed=42)...")
    keys = ["p_max_low_kw", "p_max_high_kw", "p_max_v_low_mps", "p_max_v_high_mps", "lift_off_regen_frac"]
    bounds = [(55.0, 80.0), (28.0, 50.0), (11.0, 16.5), (19.0, 26.0), (0.08, 0.22)]
    total_trials = 2 + n_lhs + n_refine

    def eval_cand(params: Dict[str, float], label: str, full_alpha_search: bool = False) -> Dict[str, Any]:
        p_lo = float(params["p_max_low_kw"])
        p_hi = min(float(params["p_max_high_kw"]), p_lo)
        v_lo = float(params["p_max_v_low_mps"])
        v_hi = max(float(params["p_max_v_high_mps"]), v_lo + 1.0)
        p_clean = {
            "p_max_low_kw": p_lo,
            "p_max_high_kw": p_hi,
            "p_max_v_low_mps": v_lo,
            "p_max_v_high_mps": v_hi,
            "lift_off_regen_frac": float(params["lift_off_regen_frac"]),
        }
        ggv_s = _scale_ggv_for_power_curve(ggv_auto, p_lo, p_hi, v_lo, v_hi)
        qss_s = solve_qss(rl_fsg.racing_track, ggv_s)
        mo = {"efficiency": p_clean}
        if full_alpha_search:
            res, _ = search_fastest_valid_lap(
                label="tv4_eff", racing_line=rl_fsg, qss=qss_s,
                ecu_driving_mode=2, tv_4wd_mode=2, n_laps=1,
                use_3d_kinematics=True, mode_overrides=mo,
            )
        else:
            res = run_transient_lap(
                label="tv4_eff", racing_line=rl_fsg, qss=qss_s,
                ecu_driving_mode=2, tv_4wd_mode=2, alpha_drv=1.0, n_laps=1,
                use_3d_kinematics=True, mode_overrides=mo,
            )
        proj = project_endurance_22km(
            label=label,
            track_length_m=rl_fsg.racing_track.length,
            t_lap_s=res.lap_time_s,
            e_net_lap_kj=res.summary["e_net_kj"],
            e_regen_lap_kj=res.summary["e_regen_kj"],
        )
        p_min = float(res.summary.get("p_chg_min_kw", 0.0))
        p_max = float(res.summary.get("p_dc_max_kw", 0.0))
        j_fsg = float(proj.e_net_22km_kwh * (proj.t_22km_s ** 2))
        feasible = (
            bool(res.valid)
            and proj.e_net_22km_kwh <= 5.20
            and p_min >= -25.05
            and p_max <= 80.05
        )
        return {
            "label": label,
            "params": {k: round(float(v), 4) for k, v in p_clean.items()},
            "metrics": {
                "tr_lap_s": round(float(res.lap_time_s), 4),
                "valid": bool(res.valid),
                "t_22km_s": round(float(proj.t_22km_s), 2),
                "e_22km_kwh": round(float(proj.e_net_22km_kwh), 3),
                "e_net_kj": round(float(res.summary.get("e_net_kj", 0.0)), 2),
                "e_regen_kj": round(float(res.summary.get("e_regen_kj", 0.0)), 2),
                "max_abs_beta_deg": round(float(res.summary.get("max_abs_beta_deg", 0.0)), 3),
                "p_dc_min_kw": round(p_min, 2),
                "p_dc_max_kw": round(p_max, 2),
                "j_fsg_e_t2": round(j_fsg, 1),
            },
            "feasible": feasible,
        }

    # Flat 47.5 kW reference baseline
    flat_ref = eval_cand({
        "p_max_low_kw": 47.5, "p_max_high_kw": 47.5,
        "p_max_v_low_mps": 14.0, "p_max_v_high_mps": 23.0,
        "lift_off_regen_frac": 0.0,
    }, "flat_47.5kw_ref", full_alpha_search=True)
    _emit_progress(pf_path, {
        "stage": "efficiency", "trial": 1, "total_trials": total_trials,
        "lap_s": flat_ref["metrics"]["tr_lap_s"], "t_22km_s": flat_ref["metrics"]["t_22km_s"],
        "e_22km_kwh": flat_ref["metrics"]["e_22km_kwh"], "j_fsg": flat_ref["metrics"]["j_fsg_e_t2"],
        "max_beta_deg": flat_ref["metrics"]["max_abs_beta_deg"], "feasible": flat_ref["feasible"],
        "is_best": True, "is_flat_ref": True, "params": flat_ref["params"],
    })

    # Initial configured P_max(vx)
    init_p = {k: float(base_cfg["modes"]["efficiency"][k]) for k in keys}
    init_c = eval_cand(init_p, "init_curve", full_alpha_search=True)
    history = [flat_ref, init_c]
    best = init_c if (init_c["feasible"] and (not flat_ref["feasible"] or init_c["metrics"]["j_fsg_e_t2"] < flat_ref["metrics"]["j_fsg_e_t2"])) else flat_ref
    _emit_progress(pf_path, {
        "stage": "efficiency", "trial": len(history), "total_trials": total_trials,
        "lap_s": init_c["metrics"]["tr_lap_s"], "t_22km_s": init_c["metrics"]["t_22km_s"],
        "e_22km_kwh": init_c["metrics"]["e_22km_kwh"], "j_fsg": init_c["metrics"]["j_fsg_e_t2"],
        "max_beta_deg": init_c["metrics"]["max_abs_beta_deg"], "feasible": init_c["feasible"],
        "is_best": best is init_c, "params": init_c["params"],
    })

    for idx, vec in enumerate(_lhs_samples(bounds, n_lhs, SEED + 200)):
        p = dict(zip(keys, vec))
        cand = eval_cand(p, f"lhs_{idx}", full_alpha_search=False)
        history.append(cand)
        is_new_best = cand["feasible"] and (not best["feasible"] or cand["metrics"]["j_fsg_e_t2"] < best["metrics"]["j_fsg_e_t2"])
        if is_new_best:
            best = cand
        _emit_progress(pf_path, {
            "stage": "efficiency", "trial": len(history), "total_trials": total_trials,
            "lap_s": cand["metrics"]["tr_lap_s"], "t_22km_s": cand["metrics"]["t_22km_s"],
            "e_22km_kwh": cand["metrics"]["e_22km_kwh"], "j_fsg": cand["metrics"]["j_fsg_e_t2"],
            "max_beta_deg": cand["metrics"]["max_abs_beta_deg"], "feasible": cand["feasible"],
            "is_best": is_new_best, "params": cand["params"],
        })

    # Local refinements around best J_FSG
    rng = np.random.default_rng(SEED + 201)
    for r_i in range(n_refine):
        p_loc = {}
        for (k, (lo, hi)) in zip(keys, bounds):
            span = 0.12 * (hi - lo)
            p_loc[k] = float(np.clip(best["params"][k] + rng.uniform(-span, span), lo, hi))
        cand = eval_cand(p_loc, f"refine_{r_i}", full_alpha_search=True)
        history.append(cand)
        is_new_best = cand["feasible"] and (not best["feasible"] or cand["metrics"]["j_fsg_e_t2"] < best["metrics"]["j_fsg_e_t2"])
        if is_new_best:
            best = cand
        _emit_progress(pf_path, {
            "stage": "efficiency", "trial": len(history), "total_trials": total_trials,
            "lap_s": cand["metrics"]["tr_lap_s"], "t_22km_s": cand["metrics"]["t_22km_s"],
            "e_22km_kwh": cand["metrics"]["e_22km_kwh"], "j_fsg": cand["metrics"]["j_fsg_e_t2"],
            "max_beta_deg": cand["metrics"]["max_abs_beta_deg"], "feasible": cand["feasible"],
            "is_best": is_new_best, "params": cand["params"],
        })

    if best["label"].startswith("lhs"):
        best = eval_cand(best["params"], best["label"] + "_verified", full_alpha_search=True)
        history.append(best)

    print(
        f"  [EFFICIENCY] Flat47.5={flat_ref['metrics']['tr_lap_s']:.3f}s / {flat_ref['metrics']['e_22km_kwh']:.2f}kWh (J={flat_ref['metrics']['j_fsg_e_t2']:.0f}) | "
        f"Best P_max(vx)={best['metrics']['tr_lap_s']:.3f}s / {best['metrics']['e_22km_kwh']:.2f}kWh (J={best['metrics']['j_fsg_e_t2']:.0f}, feasible={best['feasible']})"
    )
    return {"flat_ref": flat_ref, "baseline_curve": history[1], "best": best, "history": history}


def plot_optimization_summary(sk_res: dict, ax_res: dict, eff_res: dict) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # Panel 1: Skidpad convergence
    sk_h = sk_res["history"]
    xs = np.arange(len(sk_h))
    ys = [h["metrics"]["t_lap_s"] for h in sk_h]
    cols = [C_ACCENT if h["feasible"] else C_OPEN for h in sk_h]
    axes[0].scatter(xs, ys, c=cols, s=45, zorder=3)
    axes[0].axhline(sk_res["baseline"]["metrics"]["t_lap_s"], color=C_MUTED, ls="--", lw=1.2, label=f"Baseline ({sk_res['baseline']['metrics']['t_lap_s']:.3f}s)")
    axes[0].axhline(sk_res["best"]["metrics"]["t_lap_s"], color=C_HORIZON, ls="-", lw=1.5, label=f"Optimum ({sk_res['best']['metrics']['t_lap_s']:.3f}s)")
    axes[0].set_title("SKIDPAD Map Optimization (R=9.125 m)")
    axes[0].set_xlabel("Evaluation #")
    axes[0].set_ylabel("Skidpad Lap Time [s]")
    axes[0].legend(fontsize=8)

    # Panel 2: FSG 2023 Autocross convergence
    ax_h = ax_res["history"]
    xs2 = np.arange(len(ax_h))
    ys2 = [h["metrics"]["tr_lap_s"] for h in ax_h]
    cols2 = [C_TER27 if h["feasible"] else C_OPEN for h in ax_h]
    axes[1].scatter(xs2, ys2, c=cols2, s=45, zorder=3)
    axes[1].axhline(ax_res["baseline"]["metrics"]["tr_lap_s"], color=C_MUTED, ls="--", lw=1.2, label=f"Baseline ({ax_res['baseline']['metrics']['tr_lap_s']:.3f}s)")
    axes[1].axhline(ax_res["best"]["metrics"]["tr_lap_s"], color=C_HORIZON, ls="-", lw=1.5, label=f"Optimum ({ax_res['best']['metrics']['tr_lap_s']:.3f}s)")
    axes[1].set_title("FSG 2023 AUTOCROSS Optimization (14-DOF 3D)")
    axes[1].set_xlabel("Evaluation #")
    axes[1].set_ylabel("FSG 2023 Lap Time [s]")
    axes[1].legend(fontsize=8)

    # Panel 3: Efficiency Pareto Front (T_22km vs E_22km)
    eh = eff_res["history"]
    t_vals = [h["metrics"]["t_22km_s"] for h in eh if h["label"] != "flat_47.5kw_ref"]
    e_vals = [h["metrics"]["e_22km_kwh"] for h in eh if h["label"] != "flat_47.5kw_ref"]
    feas = [h["feasible"] for h in eh if h["label"] != "flat_47.5kw_ref"]
    c_e = [C_ACCENT if f else C_MUTED for f in feas]
    axes[2].scatter(t_vals, e_vals, c=c_e, s=42, alpha=0.85, label="P_max(vx) Candidates")
    fr = eff_res["flat_ref"]["metrics"]
    be = eff_res["best"]["metrics"]
    axes[2].scatter([fr["t_22km_s"]], [fr["e_22km_kwh"]], color=C_OPEN, marker="X", s=95, zorder=5, label=f"Flat 47.5 kW (J={fr['j_fsg_e_t2']/1e6:.3f}e6)")
    axes[2].scatter([be["t_22km_s"]], [be["e_22km_kwh"]], color=C_HORIZON, marker="*", s=135, zorder=6, label=f"Best P_max(vx) (J={be['j_fsg_e_t2']/1e6:.3f}e6)")
    axes[2].axhline(5.20, color=C_OPEN, ls=":", lw=1.0, label="E_22km Limit (5.20 kWh)")
    axes[2].set_title("FSG 2023 EFFICIENCY Pareto (T_22km vs E_22km)")
    axes[2].set_xlabel("Projected 22 km Time T_22km [s]")
    axes[2].set_ylabel("Projected 22 km Energy E_22km [kWh]")
    axes[2].legend(fontsize=7.5, loc="upper right")

    out_p = OUT_DIR / "07_tv_map_optimization_pareto.png"
    save_fig(fig, out_p)
    return out_p


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-apply", action="store_true")
    parser.add_argument("--progress-file", type=str, default="")
    parser.add_argument("--task", type=str, default="all", choices=["all", "skidpad", "autocross", "efficiency"])
    parser.add_argument("--trials", type=int, default=0)
    args, _ = parser.parse_known_args()
    pf_path = Path(args.progress_file) if args.progress_file else None
    if pf_path is not None:
        pf_path.parent.mkdir(parents=True, exist_ok=True)
        pf_path.write_text("", encoding="utf-8")

    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FSG_2023_TRACK_PATH.exists():
        build_fsg_main()

    trk_fsg = Track.load_json(FSG_2023_TRACK_PATH)
    rl_fsg = optimize_racing_line(trk_fsg, car_track_width_m=1.230, safety_margin_m=0.35)
    ggv_auto = extract_or_load_ggv(label="tv4_auto", ecu_driving_mode=2, tv_4wd_mode=0, use_3d_kinematics=True)

    base_cfg = json.loads(TV4_CFG_PATH.read_text(encoding="utf-8"))
    sk_lhs = max(2, args.trials - 3) if args.trials > 0 else 10
    sk_ref = min(2, max(1, args.trials - sk_lhs - 1)) if args.trials > 0 else 4
    ax_lhs = max(2, args.trials - 3) if args.trials > 0 else 8
    ax_ref = min(2, max(1, args.trials - ax_lhs - 1)) if args.trials > 0 else 2
    ef_lhs = max(2, args.trials - 5) if args.trials > 0 else 12
    ef_ref = min(3, max(1, args.trials - ef_lhs - 2)) if args.trials > 0 else 3

    sk_res = optimize_skidpad(base_cfg, sk_lhs, sk_ref, pf_path) if args.task in ("all", "skidpad") else None
    ax_res = optimize_autocross(base_cfg, rl_fsg, ggv_auto, ax_lhs, ax_ref, pf_path) if args.task in ("all", "autocross") else None
    eff_res = optimize_efficiency_pareto(base_cfg, rl_fsg, ggv_auto, ef_lhs, ef_ref, pf_path) if args.task in ("all", "efficiency") else None

    plot_p = plot_optimization_summary(sk_res, ax_res, eff_res) if (sk_res and ax_res and eff_res) else None

    applied: Dict[str, bool] = {"skidpad": False, "autocross": False, "efficiency": False}
    if not args.no_apply:
        updated = copy.deepcopy(base_cfg)
        if sk_res and sk_res["best"]["feasible"] and sk_res["best"]["metrics"]["t_lap_s"] <= sk_res["baseline"]["metrics"]["t_lap_s"]:
            for k, v in sk_res["best"]["params"].items():
                updated["modes"]["skidpad"][k] = v
            applied["skidpad"] = True
        if ax_res and ax_res["best"]["feasible"] and ax_res["best"]["metrics"]["tr_lap_s"] <= ax_res["baseline"]["metrics"]["tr_lap_s"]:
            for k, v in ax_res["best"]["params"].items():
                updated["modes"]["autocross"][k] = v
            applied["autocross"] = True
        if eff_res and eff_res["best"]["feasible"] and eff_res["best"]["metrics"]["j_fsg_e_t2"] <= eff_res["flat_ref"]["metrics"]["j_fsg_e_t2"]:
            for k, v in eff_res["best"]["params"].items():
                updated["modes"]["efficiency"][k] = v
            applied["efficiency"] = True
        TV4_CFG_PATH.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")

    summary = {
        "seed": SEED,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "applied_to_config": applied,
        "skidpad": sk_res,
        "autocross": ax_res,
        "efficiency": eff_res,
        "plot_path": str(plot_p) if plot_p else "",
    }
    sum_p = OUT_DIR / "tv_map_optimization_summary.json"
    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best_csv = OUT_DIR / ("trace_fsg2023_tv4_eff.csv" if args.task == "efficiency" else "trace_fsg2023_tv4_auto.csv")
    _emit_progress(pf_path, {
        "done": True,
        "summary_path": str(sum_p),
        "best_trace_csv": str(best_csv) if best_csv.exists() else str(OUT_DIR / "trace_fsg2023_tv4_auto.csv"),
    })
    print(f"\n[OPT COMPLETE] elapsed={summary['elapsed_s']:.1f}s | applied={applied} -> {sum_p}")


if __name__ == "__main__":
    main()
