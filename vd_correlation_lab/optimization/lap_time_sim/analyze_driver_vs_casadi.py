#!/usr/bin/env python3
"""
Driver Coaching & Telemetry Comparison: Driver (200 Hz SimUI CSV or QSS/14-DOF CSV)
vs CasADi Human-Achievable Validated Reference JSON.

Outputs Top-5 corners with highest time loss:
  - braking_point_delta_m (m)
  - apex_speed_delta_kmh (km/h)
  - lateral_offset_delta_m (m)
  - trail_braking_quality (0..1 score + note)
  - throttle_on_delta_m (m)
  - actionable coaching_tip

Exports a clean JSON summary consumable by the SimUI LTS tab.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: List[str], default: float = 0.0) -> np.ndarray:
    for c in candidates:
        if c in df.columns:
            return df[c].to_numpy(dtype=np.float64)
    return np.full(len(df), float(default), dtype=np.float64)


def _select_best_lap(df: pd.DataFrame, lap_req: Optional[int], track_len: float) -> pd.DataFrame:
    if "lap" not in df.columns:
        return df.copy()
    laps = sorted(int(x) for x in df["lap"].dropna().unique())
    if lap_req is not None and lap_req in laps:
        return df[df["lap"] == lap_req].copy()
    t_col = "t_s" if "t_s" in df.columns else ("t" if "t" in df.columns else None)
    best_lap = None
    best_dt = 1e9
    for lp in laps:
        if lp < 1:
            continue
        sub = df[df["lap"] == lp]
        if len(sub) < 50:
            continue
        if "s_m" in sub.columns:
            span = float(sub["s_m"].max() - sub["s_m"].min())
            if span < 0.70 * track_len:
                continue
        if t_col is not None:
            dt = float(sub[t_col].iloc[-1] - sub[t_col].iloc[0])
            if 20.0 < dt < best_dt:
                best_dt = dt
                best_lap = lp
        elif best_lap is None:
            best_lap = lp
    if best_lap is not None:
        return df[df["lap"] == best_lap].copy()
    return df[df["lap"] == laps[-1]].copy() if laps else df.copy()


def _project_xy_to_ref(
    px: np.ndarray,
    py: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_s: np.ndarray,
    ref_yaw: np.ndarray,
    track_len: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project driver (x, y) onto CasADi reference waypoints -> (s_proj, ey_signed)."""
    n_ref = len(ref_x)
    s_out = np.zeros(len(px), dtype=np.float64)
    ey_out = np.zeros(len(px), dtype=np.float64)
    prev = -1
    for i in range(len(px)):
        if prev < 0:
            d2 = (px[i] - ref_x) ** 2 + (py[i] - ref_y) ** 2
            idx = int(np.argmin(d2))
        else:
            cand = (prev + np.arange(-25, 45)) % n_ref
            d2 = (px[i] - ref_x[cand]) ** 2 + (py[i] - ref_y[cand]) ** 2
            idx = int(cand[int(np.argmin(d2))])
        prev = idx
        tx = math.cos(ref_yaw[idx])
        ty = math.sin(ref_yaw[idx])
        dx = px[i] - ref_x[idx]
        dy = py[i] - ref_y[idx]
        ds_loc = dx * tx + dy * ty
        s_out[i] = (ref_s[idx] + ds_loc) % track_len
        ey_out[i] = -dx * ty + dy * tx
    return s_out, ey_out


def _clean_monotonic_s(s_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """Return boolean mask keeping the main monotonic lap pass."""
    keep = np.ones(len(s_arr), dtype=bool)
    for k in range(1, len(s_arr)):
        if s_arr[k] < s_arr[k - 1] - 50.0:
            if k < max(15, int(0.05 * len(s_arr))):
                keep[:k] = False
            else:
                keep[k:] = False
                break
    return keep


def _interp_periodic(s_grid: np.ndarray, s_src: np.ndarray, y_src: np.ndarray, track_len: float) -> np.ndarray:
    order = np.argsort(s_src)
    s_u, uniq = np.unique(s_src[order], return_index=True)
    y_u = y_src[order][uniq]
    return np.interp(s_grid, np.append(s_u, track_len), np.append(y_u, y_u[0]))


def _compute_trail_braking_score(
    brk: np.ndarray,
    ay_g: np.ndarray,
    idx_window: np.ndarray,
) -> float:
    """
    Evaluate trail-braking quality in [0, 1] from turn-in to apex:
    rewards smooth progressive brake release while lateral acceleration builds
    (high combined-slip utilization without abrupt brake dump or brake lock).
    """
    if len(idx_window) < 2:
        return 0.5
    b_w = np.clip(brk[idx_window], 0.0, 1.0)
    ay_w = np.abs(ay_g[idx_window])
    if float(np.max(b_w)) < 0.08:
        # No braking needed in this corner
        return 1.0
    # Overlap fraction where brake is modulated (0.05..0.85) while |ay| > 0.35 g
    overlap = np.mean((b_w >= 0.05) & (ay_w >= 0.35))
    # Monotonicity of brake release (penalize double-pumping brake)
    db = np.diff(b_w)
    re_apply = float(np.sum(np.maximum(db, 0.0)))
    smooth_term = max(0.0, 1.0 - 0.8 * re_apply)
    # Combined overlap + smoothness
    score = 0.55 * min(1.0, float(overlap) * 1.8) + 0.45 * smooth_term
    return float(np.clip(score, 0.0, 1.0))


def _arc_window_indices(s_grid: np.ndarray, s_start: float, s_end: float, track_len: float) -> np.ndarray:
    s0 = s_start % track_len
    s1 = s_end % track_len
    if s0 <= s1:
        return np.where((s_grid >= s0) & (s_grid <= s1))[0]
    i1 = np.where(s_grid >= s0)[0]
    i2 = np.where(s_grid <= s1)[0]
    return np.concatenate([i1, i2])


def _find_brake_onset_s(
    s_grid: np.ndarray,
    brk: np.ndarray,
    ax_g: np.ndarray,
    s_apex: float,
    track_len: float,
    lookback_m: float = 70.0,
) -> Optional[float]:
    n = len(s_grid)
    i_apx = int(np.argmin(np.abs(s_grid - s_apex)))
    ds = track_len / float(n)
    max_steps = int(round(lookback_m / max(ds, 0.5)))
    in_brk = False
    i_onset = None
    for step in range(0, max_steps + 1):
        j = (i_apx - step) % n
        braking_now = (brk[j] >= 0.12) or (ax_g[j] <= -0.35)
        if braking_now:
            in_brk = True
            i_onset = j
        elif in_brk:
            break
    return float(s_grid[i_onset]) if i_onset is not None else None


def _find_throttle_on_s(
    s_grid: np.ndarray,
    thr: np.ndarray,
    s_apex: float,
    track_len: float,
    lookahead_m: float = 55.0,
) -> Optional[float]:
    n = len(s_grid)
    i_apx = int(np.argmin(np.abs(s_grid - s_apex)))
    ds = track_len / float(n)
    max_steps = int(round(lookahead_m / max(ds, 0.5)))
    for step in range(0, max_steps + 1):
        j = (i_apx + step) % n
        if thr[j] >= 0.70:
            return float(s_grid[j])
    return None


def _signed_arc_delta(s_drv: float, s_ref: float, track_len: float) -> float:
    d = (s_drv - s_ref) % track_len
    if d > 0.5 * track_len:
        d -= track_len
    return float(d)


def analyze_driver_vs_casadi(
    driver_log_path: Path,
    casadi_ref_path: Path,
    lap_req: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    ref_data = json.loads(casadi_ref_path.read_text(encoding="utf-8"))
    meta = ref_data["meta"]
    wps = ref_data["waypoints"]
    corners = ref_data.get("corners", [])

    n_ref = len(wps)
    ref_s = np.array([float(w["s"]) for w in wps], dtype=np.float64)
    ref_x = np.array([float(w["x"]) for w in wps], dtype=np.float64)
    ref_y = np.array([float(w["y"]) for w in wps], dtype=np.float64)
    ref_yaw = np.array([float(w["yaw"]) for w in wps], dtype=np.float64)
    ref_v = np.array([float(w["v_ref"]) for w in wps], dtype=np.float64)
    ref_thr = np.array([float(w["throttle"]) for w in wps], dtype=np.float64)
    ref_brk = np.array([float(w["brake"]) for w in wps], dtype=np.float64)
    ref_ay = np.array([float(w["ay_g"]) for w in wps], dtype=np.float64)
    ref_ax = np.array([float(w["ax_g"]) for w in wps], dtype=np.float64)
    ref_p = np.array([float(w["p_dc_kw"]) for w in wps], dtype=np.float64)

    track_len = float(ref_data.get("length_m", ref_s[-1] + (ref_s[1] - ref_s[0])))
    ds_ref = track_len / float(n_ref)
    t_ref_lap = float(meta["sim14dof_lap_time_s"] if meta.get("sim14dof_lap_time_s") is not None else meta["ocp_lap_time_s"])

    # Cumulative time of reference along s_ref
    dt_ref_seg = ds_ref / np.maximum(0.5 * (ref_v + np.roll(ref_v, -1)), 1.0)
    # Scale slightly to match exact t_ref_lap
    dt_ref_seg *= t_ref_lap / max(float(np.sum(dt_ref_seg)), 1e-6)
    t_ref_cum = np.concatenate([[0.0], np.cumsum(dt_ref_seg[:-1])])

    # Load driver log (CSV or JSON)
    has_xy = False
    if driver_log_path.suffix.lower() == ".json":
        drv_json = json.loads(driver_log_path.read_text(encoding="utf-8"))
        pts = drv_json.get("waypoints", drv_json.get("points", []))
        drv_s_raw = np.array([float(p.get("s", p.get("s_m", 0.0))) for p in pts], dtype=np.float64)
        drv_len = float(drv_json.get("length_m", drv_s_raw[-1] + (drv_s_raw[1] - drv_s_raw[0])))
        drv_s_raw = drv_s_raw * (track_len / max(drv_len, 1.0))
        drv_v_raw = np.array([float(p.get("v_ref", p.get("v_mps", 10.0))) for p in pts], dtype=np.float64)
        drv_thr_raw = np.array([float(p.get("throttle", 0.0)) for p in pts], dtype=np.float64)
        drv_brk_raw = np.array([float(p.get("brake", 0.0)) for p in pts], dtype=np.float64)
        drv_ay_raw = np.array([float(p.get("ay_g", float(p.get("ay_mps2", 0.0)) / 9.81)) for p in pts], dtype=np.float64)
        drv_ax_raw = np.array([float(p.get("ax_g", float(p.get("ax_mps2", 0.0)) / 9.81)) for p in pts], dtype=np.float64)
        drv_p_raw = np.array([float(p.get("p_dc_kw", 0.0)) for p in pts], dtype=np.float64)
        t_drv_lap = float(drv_json.get("lap_time_s", drv_json.get("meta", {}).get("sim14dof_lap_time_s", 0.0)))
        if "x" in pts[0] and "y" in pts[0]:
            has_xy = True
            px = np.array([float(p["x"]) for p in pts], dtype=np.float64)
            py = np.array([float(p["y"]) for p in pts], dtype=np.float64)
            drv_s_proj, drv_ey_raw = _project_xy_to_ref(px, py, ref_x, ref_y, ref_s, ref_yaw, track_len)
        else:
            drv_s_proj = drv_s_raw
            drv_ey_raw = np.zeros_like(drv_s_raw)
        drv_v = _interp_periodic(ref_s, drv_s_proj, drv_v_raw, track_len)
        drv_thr = np.clip(_interp_periodic(ref_s, drv_s_proj, drv_thr_raw, track_len), 0.0, 1.0)
        drv_brk = np.clip(_interp_periodic(ref_s, drv_s_proj, drv_brk_raw, track_len), 0.0, 1.0)
        drv_ay = _interp_periodic(ref_s, drv_s_proj, drv_ay_raw, track_len)
        drv_ax = _interp_periodic(ref_s, drv_s_proj, drv_ax_raw, track_len)
        drv_p = _interp_periodic(ref_s, drv_s_proj, drv_p_raw, track_len)
        drv_ey = _interp_periodic(ref_s, drv_s_proj, drv_ey_raw, track_len)
        dt_drv_seg = ds_ref / np.maximum(0.5 * (drv_v + np.roll(drv_v, -1)), 1.0)
        if t_drv_lap > 1.0:
            dt_drv_seg *= t_drv_lap / max(float(np.sum(dt_drv_seg)), 1e-6)
        else:
            t_drv_lap = float(np.sum(dt_drv_seg))
        t_drv_cum = np.concatenate([[0.0], np.cumsum(dt_drv_seg[:-1])])
    else:
        df_all = pd.read_csv(driver_log_path)
        df = _select_best_lap(df_all, lap_req, track_len)
        t_raw = _pick_col(df, ["t_s", "t"])
        t_raw = t_raw - t_raw[0]
        t_drv_lap = float(t_raw[-1]) if len(t_raw) > 1 else 0.0
        vx_raw = _pick_col(df, ["vx"])
        thr_raw = _pick_col(df, ["throttle", "thr"])
        brk_raw = _pick_col(df, ["brake", "brk"])
        ay_raw = _pick_col(df, ["ay"]) / 9.81
        ax_raw = _pick_col(df, ["ax"]) / 9.81
        p_raw = _pick_col(df, ["p_dc_kw", "P_dc_kw"])

        if "x" in df.columns and "y" in df.columns:
            has_xy = True
            px = df["x"].to_numpy(dtype=np.float64)
            py = df["y"].to_numpy(dtype=np.float64)
            s_proj, ey_raw = _project_xy_to_ref(px, py, ref_x, ref_y, ref_s, ref_yaw, track_len)
        else:
            s_raw = _pick_col(df, ["s_m", "s"])
            s_max = float(np.max(s_raw)) if len(s_raw) else track_len
            s_proj = s_raw * (track_len / max(s_max, 1.0)) if abs(s_max - track_len) > 5.0 else s_raw
            ey_raw = _pick_col(df, ["ey_race_m", "ey_base_m"], default=0.0)

        keep = _clean_monotonic_s(s_proj, t_raw)
        s_c = s_proj[keep]
        t_c = t_raw[keep] - t_raw[keep][0]
        drv_v = _interp_periodic(ref_s, s_c, vx_raw[keep], track_len)
        drv_thr = np.clip(_interp_periodic(ref_s, s_c, thr_raw[keep], track_len), 0.0, 1.0)
        drv_brk = np.clip(_interp_periodic(ref_s, s_c, brk_raw[keep], track_len), 0.0, 1.0)
        drv_ay = _interp_periodic(ref_s, s_c, ay_raw[keep], track_len)
        drv_ax = _interp_periodic(ref_s, s_c, ax_raw[keep], track_len)
        drv_p = _interp_periodic(ref_s, s_c, p_raw[keep], track_len)
        drv_ey = _interp_periodic(ref_s, s_c, ey_raw[keep], track_len)

        # Cumulative time from actual driver timestamps
        order = np.argsort(s_c)
        s_u, uniq = np.unique(s_c[order], return_index=True)
        t_u = t_c[order][uniq]
        t_drv_cum = np.interp(ref_s, np.append(s_u, track_len), np.append(t_u, t_drv_lap))
        dt_drv_seg = np.diff(np.append(t_drv_cum, t_drv_lap))

    # Energy 22 km estimate for driver
    e_drv_kj = float(np.sum(drv_p * dt_drv_seg))
    e_drv_22km = (e_drv_kj / 3600.0) * (22000.0 / max(track_len, 1.0))
    delta_t_cum = t_drv_cum - t_ref_cum

    # Per-corner analysis
    corner_reports: List[Dict[str, Any]] = []
    for c in corners:
        cid = int(c["corner_id"])
        cname = str(c["name"])
        s_start = float(c["s_start"])
        s_brake_ref = float(c["s_brake"])
        s_turn_ref = float(c["s_turn_in"])
        s_apex_ref = float(c["s_apex"])
        s_exit_ref = float(c["s_exit"])

        idx_corner = _arc_window_indices(ref_s, s_start, s_exit_ref, track_len)
        if len(idx_corner) < 2:
            continue
        t_drv_c = float(np.sum(dt_drv_seg[idx_corner]))
        t_ref_c = float(np.sum(dt_ref_seg[idx_corner]))
        time_loss_s = t_drv_c - t_ref_c

        # Braking point delta
        s_brk_drv = _find_brake_onset_s(ref_s, drv_brk, drv_ax, s_apex_ref, track_len)
        s_brk_ref_det = _find_brake_onset_s(ref_s, ref_brk, ref_ax, s_apex_ref, track_len)
        if s_brk_drv is not None and s_brk_ref_det is not None:
            brk_delta_m = _signed_arc_delta(s_brk_drv, s_brk_ref_det, track_len)
        else:
            brk_delta_m = 0.0

        # Apex min speed delta
        idx_apx_win = _arc_window_indices(ref_s, s_turn_ref, s_exit_ref, track_len)
        v_apx_drv_kmh = float(np.min(drv_v[idx_apx_win]) * 3.6) if len(idx_apx_win) else float(c["v_apex_kmh"])
        v_apx_ref_kmh = float(np.min(ref_v[idx_apx_win]) * 3.6) if len(idx_apx_win) else float(c["v_apex_kmh"])
        apex_v_delta_kmh = v_apx_drv_kmh - v_apx_ref_kmh

        # Lateral offset RMS delta
        lat_off_m = float(np.sqrt(np.mean(drv_ey[idx_corner] ** 2))) if (has_xy or np.any(drv_ey != 0.0)) else 0.0

        # Trail-braking quality (turn-in to apex)
        idx_tb = _arc_window_indices(ref_s, s_turn_ref, s_apex_ref, track_len)
        tb_drv = _compute_trail_braking_score(drv_brk, drv_ay, idx_tb)
        tb_ref = _compute_trail_braking_score(ref_brk, ref_ay, idx_tb)
        if tb_drv >= 0.78:
            tb_note = "Clean progressive trail-braking"
        elif tb_drv >= 0.52:
            tb_note = "Moderate trail-braking; release brake more smoothly toward apex"
        else:
            tb_note = "Abrupt brake release or early off-brake before turn-in"

        # Throttle-on delta
        s_thr_drv = _find_throttle_on_s(ref_s, drv_thr, s_apex_ref, track_len)
        s_thr_ref = _find_throttle_on_s(ref_s, ref_thr, s_apex_ref, track_len)
        if s_thr_drv is not None and s_thr_ref is not None:
            thr_delta_m = _signed_arc_delta(s_thr_drv, s_thr_ref, track_len)
        else:
            thr_delta_m = 0.0

        # Build concise actionable coaching tip
        tips: List[str] = []
        if brk_delta_m < -3.0:
            tips.append(f"Brake {abs(brk_delta_m):.1f} m later")
        elif brk_delta_m > 3.5:
            tips.append(f"Brake {brk_delta_m:.1f} m earlier to protect apex")
        if apex_v_delta_kmh < -2.0:
            tips.append(f"carry +{abs(apex_v_delta_kmh):.1f} km/h more apex speed")
        if lat_off_m > 0.25:
            tips.append(f"tighten line by {lat_off_m:.2f} m RMS")
        if thr_delta_m > 3.0:
            tips.append(f"get on throttle {thr_delta_m:.1f} m earlier")
        if tb_drv < 0.60 and tb_ref > 0.70:
            tips.append("blend trail-braking deeper into apex")
        if not tips:
            tips.append("Focus on smooth minimum-speed roll and earlier exit commit")

        corner_reports.append({
            "corner_id": cid,
            "name": cname,
            "s_start": round(s_start, 2),
            "s_brake": round(s_brake_ref, 2),
            "s_apex": round(s_apex_ref, 2),
            "s_exit": round(s_exit_ref, 2),
            "time_loss_s": round(float(time_loss_s), 4),
            "braking_point_delta_m": round(float(brk_delta_m), 2),
            "apex_speed_driver_kmh": round(v_apx_drv_kmh, 2),
            "apex_speed_ref_kmh": round(v_apx_ref_kmh, 2),
            "apex_speed_delta_kmh": round(float(apex_v_delta_kmh), 2),
            "lateral_offset_delta_m": round(float(lat_off_m), 3),
            "trail_braking_quality": round(float(tb_drv), 3),
            "trail_braking_ref_quality": round(float(tb_ref), 3),
            "trail_braking_note": tb_note,
            "throttle_on_delta_m": round(float(thr_delta_m), 2),
            "coaching_tip": "; ".join(tips),
        })

    corner_reports_sorted = sorted(corner_reports, key=lambda x: x["time_loss_s"], reverse=True)
    top_corners = corner_reports_sorted[: max(1, int(top_k))]

    return {
        "summary": {
            "driver_log": str(driver_log_path),
            "casadi_reference": str(casadi_ref_path),
            "track_name": str(meta.get("track_name", "fsg_2023")),
            "grip_profile": str(meta.get("grip_profile", "fsg_real")),
            "driver_lap_time_s": round(t_drv_lap, 4),
            "casadi_lap_time_s": round(t_ref_lap, 4),
            "total_time_loss_s": round(t_drv_lap - t_ref_lap, 4),
            "total_time_loss_pct": round(100.0 * (t_drv_lap - t_ref_lap) / max(t_ref_lap, 1e-6), 2),
            "v_max_driver_kmh": round(float(np.max(drv_v) * 3.6), 2),
            "v_max_ref_kmh": round(float(np.max(ref_v) * 3.6), 2),
            "ay_max_driver_g": round(float(np.max(np.abs(drv_ay))), 3),
            "ay_max_ref_g": round(float(np.max(np.abs(ref_ay))), 3),
            "e_22km_driver_kwh": round(e_drv_22km, 3),
            "e_22km_ref_kwh": round(float(meta.get("e_22km_kwh", 0.0)), 3),
            "top_corners_time_loss_s": round(float(sum(c["time_loss_s"] for c in top_corners)), 4),
        },
        "top_corners": top_corners,
        "all_corners": corner_reports,
        "distance_series": {
            "s_m": [round(float(v), 2) for v in ref_s],
            "delta_t_s": [round(float(v), 4) for v in delta_t_cum],
            "v_driver_kmh": [round(float(v * 3.6), 2) for v in drv_v],
            "v_ref_kmh": [round(float(v * 3.6), 2) for v in ref_v],
            "thr_driver": [round(float(v), 3) for v in drv_thr],
            "thr_ref": [round(float(v), 3) for v in ref_thr],
            "brk_driver": [round(float(v), 3) for v in drv_brk],
            "brk_ref": [round(float(v), 3) for v in ref_brk],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Driver/QSS CSV vs CasADi Human-Achievable Reference JSON")
    parser.add_argument("--driver-log", type=str, required=True, help="Path to 200Hz SimUI CSV, 14-DOF CSV, or QSS JSON")
    parser.add_argument("--casadi-ref", type=str, required=True, help="Path to CasADi validated reference JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to write coaching JSON summary")
    parser.add_argument("--lap", type=int, default=None, help="Optional lap number to analyze")
    parser.add_argument("--top-k", type=int, default=5, help="Number of highest time-loss corners (default: 5)")
    args = parser.parse_args()

    res = analyze_driver_vs_casadi(
        driver_log_path=Path(args.driver_log),
        casadi_ref_path=Path(args.casadi_ref),
        lap_req=args.lap,
        top_k=args.top_k,
    )
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")

    s = res["summary"]
    print(
        f"[COACHING] Driver={s['driver_lap_time_s']:.3f}s vs CasADi={s['casadi_lap_time_s']:.3f}s "
        f"(loss={s['total_time_loss_s']:+.3f}s, {s['total_time_loss_pct']:+.2f}%)",
        flush=True,
    )
    for rank, c in enumerate(res["top_corners"], start=1):
        print(
            f"  #{rank} {c['name']:12s}: loss={c['time_loss_s']:+.3f}s | "
            f"d_brk={c['braking_point_delta_m']:+.1f}m | d_apex={c['apex_speed_delta_kmh']:+.1f}km/h | "
            f"lat_rms={c['lateral_offset_delta_m']:.2f}m | tb={c['trail_braking_quality']:.2f} | "
            f"d_thr={c['throttle_on_delta_m']:+.1f}m | {c['coaching_tip']}",
            flush=True,
        )
    print(f"[COACHING] Saved -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
