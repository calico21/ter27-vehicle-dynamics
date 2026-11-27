#!/usr/bin/env python3
"""
3D Suspension Hardpoint Load Envelopes & ASTM E1049-85 Rainflow Fatigue Spectra
Exporter for ANSYS Mechanical (TeR27 4WD Double-Wishbone + Pushrod + Rocker).

Computes exact 6-DOF upright static equilibrium across:
  1. Full 200 Hz FSG 2023 Max Attack & Endurance telemetry (`trace_casadi_fsg_real.csv`)
  2. Standard FSAE extreme structural design load cases (3G Bump, 2G Brake,
     2G Cornering, Combined 1.5G Brake+Corner, 4G Curb Strike)

Outputs in `outputs/validation_vault/06_suspension_kinematics_and_fea/`:
  - fea_hardpoint_loads.json
  - fea_hardpoint_loads.csv (ANSYS Mechanical boundary-condition table)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vd_correlation_lab.config.paths import (  # noqa: E402
    CASADI_SUITE_DIR,
    TER27_CONFIG_DIR,
    VAULT_06_KINEMATICS_FEA,
    ensure_vault_dirs,
)


MEMBER_NAMES: List[str] = [
    "lca_front",
    "lca_rear",
    "uca_front",
    "uca_rear",
    "tie_rod",
    "pushrod",
]

MEMBER_LABELS: Dict[str, str] = {
    "lca_front": "LCA Front Leg",
    "lca_rear": "LCA Rear Leg",
    "uca_front": "UCA Front Leg",
    "uca_rear": "UCA Rear Leg",
    "tie_rod": "Tie-Rod / Toe-Link",
    "pushrod": "Pushrod",
    "rocker_pivot": "Rocker Pivot Bearing",
    "damper_mount": "Damper / Coilover Mount",
    "arb_link": "ARB Drop-Link",
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3, dtype=np.float64)


def build_corner_equilibrium_matrix(hp: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build exact 6x6 static equilibrium matrix A mapping link axial forces
    F_links = [F_lca_f, F_lca_r, F_uca_f, F_uca_r, F_tie, F_push]^T (N, + = tension)
    to upright resultant [Fx, Fy, Fz, Mx, My, Mz]^T about wheel center.
    """
    p_wc = np.asarray(hp["wheel_center"], dtype=np.float64)
    p_cp = np.asarray(hp["contact_patch"], dtype=np.float64)
    p_lbj = np.asarray(hp["lbj"], dtype=np.float64)
    p_ubj = np.asarray(hp["ubj"], dtype=np.float64)
    p_lca_f = np.asarray(hp["lca_front"], dtype=np.float64)
    p_lca_r = np.asarray(hp["lca_rear"], dtype=np.float64)
    p_uca_f = np.asarray(hp["uca_front"], dtype=np.float64)
    p_uca_r = np.asarray(hp["uca_rear"], dtype=np.float64)
    p_tie_o = np.asarray(hp["tie_outer"], dtype=np.float64)
    p_tie_i = np.asarray(hp["tie_inner"], dtype=np.float64)
    p_pr_w = np.asarray(hp["pushrod_wishbone"], dtype=np.float64)
    p_pr_r = np.asarray(hp["pushrod_rocker"], dtype=np.float64)

    # Unit vectors pointing from upright/wishbone outboard joint toward chassis/rocker
    links = [
        (p_lbj, _unit(p_lca_f - p_lbj)),
        (p_lbj, _unit(p_lca_r - p_lbj)),
        (p_ubj, _unit(p_uca_f - p_ubj)),
        (p_ubj, _unit(p_uca_r - p_ubj)),
        (p_tie_o, _unit(p_tie_i - p_tie_o)),
        (p_pr_w, _unit(p_pr_r - p_pr_w)),
    ]

    A = np.zeros((6, 6), dtype=np.float64)
    for j, (p_att, u_vec) in enumerate(links):
        r = p_att - p_wc
        A[0:3, j] = u_vec
        A[3:6, j] = np.cross(r, u_vec)

    # Rocker geometry unit vectors
    u_push_on_rocker = _unit(p_pr_w - p_pr_r)
    u_dmp = _unit(np.asarray(hp["damper_chassis"], dtype=np.float64) - np.asarray(hp["damper_rocker"], dtype=np.float64))
    u_arb = _unit(np.asarray(hp["arb_arm_tip"], dtype=np.float64) - np.asarray(hp["arb_link_rocker"], dtype=np.float64))

    aux = {
        "r_cp": p_cp - p_wc,
        "u_push_on_rocker": u_push_on_rocker,
        "u_dmp": u_dmp,
        "u_arb": u_arb,
        "cond_A": float(np.linalg.cond(A)),
    }
    return A, aux


def solve_link_forces_batch(
    A_inv: np.ndarray,
    r_cp: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    mz_tire: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve 6 link axial forces (N, +tension / -compression) for N timesteps.
    External tire load F_ext = [fx, fy, fz] applied at contact patch r_cp.
    Equilibrium: A * F_links + W_ext = 0 => F_links = -A_inv @ W_ext.
    """
    n = fx.shape[0]
    W = np.zeros((6, n), dtype=np.float64)
    W[0, :] = fx
    W[1, :] = fy
    W[2, :] = fz
    # Moment about wheel center: r_cp x F_ext
    W[3, :] = r_cp[1] * fz - r_cp[2] * fy
    W[4, :] = r_cp[2] * fx - r_cp[0] * fz
    W[5, :] = r_cp[0] * fy - r_cp[1] * fx
    if mz_tire is not None:
        W[5, :] += mz_tire
    return -(A_inv @ W)  # shape (6, n)


def rainflow_count_4point(signal: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    ASTM E1049-85 4-point Rainflow Cycle Counting algorithm.
    Returns list of (range_N, mean_N, count) where count is 1.0 (full) or 0.5 (half).
    """
    # 1. Extract turning points (local extrema)
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 3:
        return []
    dx = np.diff(x)
    # Remove consecutive duplicates
    keep = np.concatenate([[True], np.abs(dx) > 1e-9])
    x = x[keep]
    if x.size < 3:
        return []
    d1 = np.diff(x)
    turns_idx = np.where(d1[:-1] * d1[1:] < 0.0)[0] + 1
    pts = np.concatenate([[x[0]], x[turns_idx], [x[-1]]])

    # 2. 4-point stack extraction
    stack: List[float] = []
    cycles: List[Tuple[float, float, float]] = []
    for p in pts:
        stack.append(float(p))
        while len(stack) >= 4:
            x0, x1, x2, x3 = stack[-4], stack[-3], stack[-2], stack[-1]
            r1 = abs(x1 - x0)
            r2 = abs(x2 - x1)
            r3 = abs(x3 - x2)
            if r2 <= r1 and r2 <= r3:
                cycles.append((r2, 0.5 * (x1 + x2), 1.0))
                # Remove x1 and x2
                stack.pop(-2)
                stack.pop(-2)
            else:
                break

    # Remaining residuals as half-cycles
    for i in range(len(stack) - 1):
        r = abs(stack[i + 1] - stack[i])
        m = 0.5 * (stack[i + 1] + stack[i])
        cycles.append((r, m, 0.5))
    return cycles


def compute_miner_fatigue(
    cycles_per_lap: List[Tuple[float, float, float]],
    n_laps: float,
    area_mm2: float = 33.36,
    s_ut_mpa: float = 670.0,
    s_e_mpa: float = 280.0,
    kt_weld: float = 1.65,
) -> Dict[str, Any]:
    """
    Compute Goodman-corrected Basquin fatigue damage (Palmgren-Miner rule)
    for 4130 Chromoly tube (OD 12.7 mm, t 0.9 mm, A = 33.36 mm^2) with weld Kt.
    """
    # Basquin curve: S_a = a * N^b passing through (1e3, 0.9*Sut) and (1e6, Se)
    s_1e3 = 0.9 * s_ut_mpa
    b = math.log10(s_e_mpa / s_1e3) / 3.0
    a = s_1e3 / (1000.0 ** b)

    damage_lap = 0.0
    max_amp_n = 0.0
    total_cycles_lap = 0.0
    for rng, mean_f, cnt in cycles_per_lap:
        amp_n = 0.5 * rng
        max_amp_n = max(max_amp_n, amp_n)
        total_cycles_lap += cnt
        sig_a = kt_weld * amp_n / area_mm2
        sig_m = max(0.0, mean_f / area_mm2)  # tensile mean reduces life
        if sig_a <= 1e-6:
            continue
        goodman_denom = max(1.0 - sig_m / s_ut_mpa, 0.15)
        sig_eq = sig_a / goodman_denom
        if sig_eq >= 0.45 * s_e_mpa:
            n_fail = (sig_eq / a) ** (1.0 / b)
            damage_lap += cnt / max(n_fail, 1.0)

    damage_total = damage_lap * n_laps
    life_laps = (1.0 / damage_lap) if damage_lap > 1e-15 else 1e9
    return {
        "cycles_per_lap": round(total_cycles_lap, 1),
        "max_amplitude_n": round(max_amp_n, 2),
        "miner_damage_22km": float(f"{damage_total:.4e}"),
        "estimated_life_laps": round(min(life_laps, 1e6), 1),
        "estimated_life_km": round(min(life_laps * 1.22, 1e6), 1),
        "fatigue_pass_22km": bool(damage_total < 0.1),
    }


def export_fea_hardpoint_loads() -> Dict[str, Any]:
    """Compute and export 3D FEA hardpoint load envelopes & rainflow fatigue spectra."""
    ensure_vault_dirs()
    susp_cfg = json.loads((TER27_CONFIG_DIR / "suspension.json").read_text(encoding="utf-8"))
    veh_cfg = json.loads((TER27_CONFIG_DIR / "vehicle.json").read_text(encoding="utf-8"))
    dw = susp_cfg["double_wishbone_kinematics"]

    mass_kg = float(veh_cfg["mass"])
    g = float(veh_cfg["gravity"])
    fwd = float(veh_cfg["front_weight_distribution"])
    fz_stat_f = 0.5 * fwd * mass_kg * g
    fz_stat_r = 0.5 * (1.0 - fwd) * mass_kg * g

    # Load 200 Hz FSG Max Attack telemetry
    trace_path = CASADI_SUITE_DIR / "trace_casadi_fsg_real.csv"
    data = np.genfromtxt(trace_path, delimiter=",", names=True, dtype=np.float64)
    n_samples = int(data.shape[0])
    print(f"[FEA] Loaded {n_samples} samples from {trace_path.name}", flush=True)

    # Corner definitions: Front Left (wheel 0) and Rear Left (wheel 2)
    corners = [
        ("front_left", dw["front_left"], 0, fz_stat_f, float(susp_cfg["motion_ratio_front"])),
        ("rear_left", dw["rear_left"], 2, fz_stat_r, float(susp_cfg["motion_ratio_rear"])),
    ]

    results: Dict[str, Any] = {
        "vehicle": "TeR27 4WD",
        "coordinate_system": "ISO 8855 (+X forward, +Y left, +Z up)",
        "sign_convention": "Positive (+N) = Tension, Negative (-N) = Compression",
        "telemetry_source": str(trace_path.name),
        "n_telemetry_samples": n_samples,
        "endurance_laps_22km": 18,
        "axles": {},
    }
    csv_rows: List[Dict[str, Any]] = []

    for axle_name, hp, w_idx, fz_stat, mr in corners:
        A, aux = build_corner_equilibrium_matrix(hp)
        A_inv = np.linalg.inv(A)
        r_cp = aux["r_cp"]

        fx_lap = data[f"fx{w_idx}"]
        fy_lap = data[f"fy{w_idx}"]
        fz_lap = data[f"fz{w_idx}"]
        fspr_lap = data[f"fspr{w_idx}"]
        fdmp_lap = data[f"fdmp{w_idx}"]
        farb_lap = data[f"farb{w_idx}"]

        # Solve 6 link forces across entire FSG lap
        f_links_lap = solve_link_forces_batch(A_inv, r_cp, fx_lap, fy_lap, fz_lap)

        # Rocker reaction & coilover / ARB forces over lap
        f_push_lap = f_links_lap[5, :]
        f_damper_lap = (fspr_lap + fdmp_lap) / max(mr, 0.5)
        f_arb_lap = farb_lap / max(mr, 0.5)
        f_rocker_vec = (
            np.outer(aux["u_push_on_rocker"], f_push_lap)
            + np.outer(aux["u_dmp"], f_damper_lap)
            + np.outer(aux["u_arb"], f_arb_lap)
        )
        f_rocker_mag_lap = np.linalg.norm(f_rocker_vec, axis=0)

        # FSAE Standard Extreme Static Load Cases (per corner)
        m_c = fz_stat / g
        extreme_cases = {
            "3G_BUMP": (0.0, 0.0, 3.0 * fz_stat),
            "2G_BRAKE": (-2.0 * m_c * g * (1.25 if w_idx == 0 else 0.75), 0.0, 1.65 * fz_stat if w_idx == 0 else 0.65 * fz_stat),
            "2G_CORNER_OUTSIDE": (0.0, -2.0 * m_c * g, 1.85 * fz_stat),
            "2G_CORNER_INSIDE": (0.0, 2.0 * m_c * g, 0.45 * fz_stat),
            "COMBINED_1.5G_BRK_CORN": (-1.5 * m_c * g, -1.5 * m_c * g, 2.20 * fz_stat),
            "CURB_STRIKE_4G": (-0.8 * fz_stat, -0.8 * fz_stat, 4.0 * fz_stat),
        }

        ext_fx = np.array([v[0] for v in extreme_cases.values()])
        ext_fy = np.array([v[1] for v in extreme_cases.values()])
        ext_fz = np.array([v[2] for v in extreme_cases.values()])
        f_links_ext = solve_link_forces_batch(A_inv, r_cp, ext_fx, ext_fy, ext_fz)

        case_names = list(extreme_cases.keys())
        members_dict: Dict[str, Any] = {}
        rainflow_spectra: Dict[str, Any] = {}

        for m_idx, m_name in enumerate(MEMBER_NAMES):
            sig = f_links_lap[m_idx, :]
            ext_vals = f_links_ext[m_idx, :]
            lap_max_t = float(np.max(sig))
            lap_max_c = float(np.min(sig))
            ext_max_t = float(np.max(ext_vals))
            ext_max_c = float(np.min(ext_vals))
            peak_abs = max(abs(lap_max_t), abs(lap_max_c), abs(ext_max_t), abs(ext_max_c))
            gov_idx = int(np.argmax(np.abs(ext_vals)))

            # Rainflow fatigue over 1 lap -> 18 laps (22 km)
            cycles = rainflow_count_4point(sig)
            fatigue = compute_miner_fatigue(cycles, n_laps=18.0)

            # Histogram bins for plotting (12 bins from 0 to max range)
            rngs = np.array([c[0] for c in cycles], dtype=np.float64) if cycles else np.array([0.0])
            cnts = np.array([c[2] * 18.0 for c in cycles], dtype=np.float64) if cycles else np.array([0.0])
            hist_cnts, hist_edges = np.histogram(rngs, bins=14, weights=cnts)

            members_dict[m_name] = {
                "label": MEMBER_LABELS[m_name],
                "lap_max_tension_n": round(lap_max_t, 2),
                "lap_max_compression_n": round(lap_max_c, 2),
                "lap_rms_n": round(float(np.sqrt(np.mean(sig ** 2))), 2),
                "extreme_max_tension_n": round(ext_max_t, 2),
                "extreme_max_compression_n": round(ext_max_c, 2),
                "governing_load_case": case_names[gov_idx],
                "design_peak_abs_n": round(peak_abs, 2),
                "design_peak_with_sf15_n": round(peak_abs * 1.5, 2),
                "fatigue_22km": fatigue,
            }
            rainflow_spectra[m_name] = {
                "bin_edges_n": [round(float(x), 2) for x in hist_edges],
                "cycles_22km": [round(float(x), 2) for x in hist_cnts],
            }
            csv_rows.append({
                "axle": axle_name,
                "member": m_name,
                "label": MEMBER_LABELS[m_name],
                "lap_max_tension_N": round(lap_max_t, 2),
                "lap_max_compression_N": round(lap_max_c, 2),
                "extreme_max_tension_N": round(ext_max_t, 2),
                "extreme_max_compression_N": round(ext_max_c, 2),
                "design_peak_abs_N": round(peak_abs, 2),
                "ansys_design_load_SF1p5_N": round(peak_abs * 1.5, 2),
                "governing_case": case_names[gov_idx],
                "miner_damage_22km": fatigue["miner_damage_22km"],
            })

        # Add Rocker Pivot, Damper Mount, ARB Link summary
        ext_push = f_links_ext[5, :]
        ext_dmp = (ext_fz - fz_stat) / max(mr, 0.5)
        ext_rocker_mag = np.linalg.norm(
            np.outer(aux["u_push_on_rocker"], ext_push) + np.outer(aux["u_dmp"], ext_dmp),
            axis=0,
        )
        for aux_name, lap_arr, ext_arr in [
            ("rocker_pivot", f_rocker_mag_lap, ext_rocker_mag),
            ("damper_mount", f_damper_lap, ext_dmp),
            ("arb_link", f_arb_lap, 0.35 * ext_dmp),
        ]:
            peak_a = max(float(np.max(np.abs(lap_arr))), float(np.max(np.abs(ext_arr))))
            members_dict[aux_name] = {
                "label": MEMBER_LABELS[aux_name],
                "lap_max_tension_n": round(float(np.max(lap_arr)), 2),
                "lap_max_compression_n": round(float(np.min(lap_arr)), 2),
                "lap_rms_n": round(float(np.sqrt(np.mean(lap_arr ** 2))), 2),
                "extreme_max_tension_n": round(float(np.max(ext_arr)), 2),
                "extreme_max_compression_n": round(float(np.min(ext_arr)), 2),
                "governing_load_case": case_names[int(np.argmax(np.abs(ext_arr)))],
                "design_peak_abs_n": round(peak_a, 2),
                "design_peak_with_sf15_n": round(peak_a * 1.5, 2),
            }
            csv_rows.append({
                "axle": axle_name,
                "member": aux_name,
                "label": MEMBER_LABELS[aux_name],
                "lap_max_tension_N": round(float(np.max(lap_arr)), 2),
                "lap_max_compression_N": round(float(np.min(lap_arr)), 2),
                "extreme_max_tension_N": round(float(np.max(ext_arr)), 2),
                "extreme_max_compression_N": round(float(np.min(ext_arr)), 2),
                "design_peak_abs_N": round(peak_a, 2),
                "ansys_design_load_SF1p5_N": round(peak_a * 1.5, 2),
                "governing_case": case_names[int(np.argmax(np.abs(ext_arr)))],
                "miner_damage_22km": 0.0,
            })

        results["axles"][axle_name] = {
            "static_corner_fz_n": round(fz_stat, 2),
            "equilibrium_matrix_cond": round(aux["cond_A"], 3),
            "members": members_dict,
            "rainflow_spectra_22km": rainflow_spectra,
        }

    # Global peak summary for quick KPI lookup
    all_peaks = [
        (axle, m, d["design_peak_abs_n"], d["governing_load_case"])
        for axle, ad in results["axles"].items()
        for m, d in ad["members"].items()
    ]
    top_peak = max(all_peaks, key=lambda x: x[2])
    results["global_peak_summary"] = {
        "max_member_axle": top_peak[0],
        "max_member": top_peak[1],
        "max_design_load_n": top_peak[2],
        "max_design_load_sf15_n": round(top_peak[2] * 1.5, 2),
        "governing_case": top_peak[3],
    }

    out_json = VAULT_06_KINEMATICS_FEA / "fea_hardpoint_loads.json"
    out_csv = VAULT_06_KINEMATICS_FEA / "fea_hardpoint_loads.csv"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[FEA] Exported {out_json} and {out_csv}", flush=True)
    return results


if __name__ == "__main__":
    export_fea_hardpoint_loads()
