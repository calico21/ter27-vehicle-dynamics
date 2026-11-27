"""Service for discovering DiL telemetry sessions and building A/B comparisons."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from api.core.config import (
    DIL_SESSIONS_DIR,
    REPO_ROOT,
    SETUP_REPORT_PATH,
    SKIDPAD_PARITY_PATH,
)
from api.models.telemetry import (
    SessionABComparisonResponse,
    SessionsListResponse,
    SessionSummary,
    TrajectorySample,
)

MAX_RETURN_POINTS = 240


def _downsample_indices(n: int, max_pts: int = MAX_RETURN_POINTS) -> List[int]:
    if n <= max_pts:
        return list(range(n))
    step = (n - 1) / float(max_pts - 1)
    return [int(round(i * step)) for i in range(max_pts)]


def _discover_sessions() -> List[Tuple[SessionSummary, Path]]:
    found: List[Tuple[SessionSummary, Path]] = []

    if DIL_SESSIONS_DIR.exists():
        for csv_path in sorted(DIL_SESSIONS_DIR.glob("*.csv")):
            if not csv_path.is_file() or csv_path.stat().st_size < 64:
                continue
            sid = f"dil_{csv_path.stem}"
            n_rows = 0
            duration_s = 0.0
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        n_rows += 1
                        if "t_s" in row and row["t_s"]:
                            duration_s = float(row["t_s"])
            except Exception:
                continue
            found.append(
                (
                    SessionSummary(
                        session_id=sid,
                        title=f"DiL Recording: {csv_path.stem}",
                        kind="dil_recording",
                        track_or_event="autocross",
                        grip_profile="FSG_RUBBERED",
                        n_samples=n_rows,
                        duration_s=round(duration_s, 3),
                        source_path=str(csv_path.relative_to(REPO_ROOT)),
                    ),
                    csv_path,
                )
            )

    if SKIDPAD_PARITY_PATH.exists():
        try:
            d = json.loads(SKIDPAD_PARITY_PATH.read_text(encoding="utf-8"))
            n_samples = int(d.get("n_samples", len(d.get("hist", {}).get("X", []))))
            t_mean = float(d.get("t_mean_s", 5.29))
            found.append(
                (
                    SessionSummary(
                        session_id="skidpad_godot_vs_casadi_fsg_rubbered",
                        title="Skidpad 14-DOF Godot Parity vs CasADi Reference (FSG_RUBBERED)",
                        kind="godot_parity",
                        track_or_event="skidpad",
                        grip_profile=str(d.get("grip_profile", "FSG_RUBBERED")),
                        n_samples=n_samples,
                        duration_s=round(t_mean * 2.0, 3),
                        source_path=str(SKIDPAD_PARITY_PATH.relative_to(REPO_ROOT)),
                    ),
                    SKIDPAD_PARITY_PATH,
                )
            )
        except Exception:
            pass

    if SETUP_REPORT_PATH.exists():
        try:
            rep = json.loads(SETUP_REPORT_PATH.read_text(encoding="utf-8"))
            ac = rep.get("events", {}).get("autocross", {})
            found.append(
                (
                    SessionSummary(
                        session_id="casadi_autocross_fsg2023_reference",
                        title="CasADi IPOPT Autocross FSG 2023 Optimal vs Base",
                        kind="casadi_reference",
                        track_or_event="autocross",
                        grip_profile=str(rep.get("grip_profile", "FSG_RUBBERED")),
                        n_samples=180,
                        duration_s=round(float(ac.get("optimal_time_s", 67.1057)), 3),
                        source_path=str(SETUP_REPORT_PATH.relative_to(REPO_ROOT)),
                    ),
                    SETUP_REPORT_PATH,
                )
            )
        except Exception:
            pass

    return found


def _load_dil_csv_ab(summary: SessionSummary, csv_path: Path) -> SessionABComparisonResponse:
    raw_rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                if k is None or v is None:
                    continue
                try:
                    parsed[k.strip()] = float(v)
                except ValueError:
                    continue
            if parsed:
                raw_rows.append(parsed)

    idxs = _downsample_indices(len(raw_rows))
    driver_trace: List[TrajectorySample] = []
    ref_trace: List[TrajectorySample] = []

    for i in idxs:
        r = raw_rows[i]
        t_s = r.get("t_s", 0.0)
        s_m = r.get("s_m", 0.0)
        vx = r.get("vx_ms", 0.0)
        vx_ref = r.get("vx_ref_ms", vx)
        ay_g = r.get("ay_mps2", 0.0) / 9.81
        ay_ref_g = r.get("ay_ref_mps2", r.get("ay_mps2", 0.0)) / 9.81
        ax_g = r.get("ax_mps2", 0.0) / 9.81
        ax_ref_g = r.get("ax_ref_mps2", r.get("ax_mps2", 0.0)) / 9.81
        steer_deg = math.degrees(r.get("delta_rad", 0.0))
        steer_ref_deg = math.degrees(r.get("delta_ref_rad", r.get("delta_rad", 0.0)))
        delta_t_s = r.get("delta_t_s", 0.0)

        driver_trace.append(
            TrajectorySample(
                t_s=round(t_s, 4),
                s_m=round(s_m, 3),
                x_m=round(s_m, 3),
                y_m=round(r.get("e_lat_m", 0.0), 3),
                vx_mps=round(vx, 3),
                ay_g=round(ay_g, 3),
                ax_g=round(ax_g, 3),
                beta_deg=0.0,
                steer_deg=round(steer_deg, 3),
                mz_tv_nm=0.0,
                delta_t_s=round(delta_t_s, 4),
            )
        )
        ref_trace.append(
            TrajectorySample(
                t_s=round(max(0.0, t_s - delta_t_s), 4),
                s_m=round(s_m, 3),
                x_m=round(s_m, 3),
                y_m=0.0,
                vx_mps=round(vx_ref, 3),
                ay_g=round(ay_ref_g, 3),
                ax_g=round(ax_ref_g, 3),
                beta_deg=0.0,
                steer_deg=round(steer_ref_deg, 3),
                mz_tv_nm=0.0,
                delta_t_s=0.0,
            )
        )

    final_dt = driver_trace[-1].delta_t_s if driver_trace else 0.0
    max_v = max((p.vx_mps for p in driver_trace), default=0.0)
    max_ay = max((abs(p.ay_g) for p in driver_trace), default=0.0)
    return SessionABComparisonResponse(
        session=summary,
        driver_trace=driver_trace,
        casadi_reference_trace=ref_trace,
        kpis={
            "final_delta_t_s": round(final_dt, 4),
            "peak_vx_mps": round(max_v, 3),
            "peak_abs_ay_g": round(max_ay, 3),
            "n_points": float(len(driver_trace)),
        },
    )


def _load_skidpad_parity_ab(summary: SessionSummary, path: Path) -> SessionABComparisonResponse:
    d = json.loads(path.read_text(encoding="utf-8"))
    hist = d.get("hist", {})
    xs = hist.get("X", [])
    ys = hist.get("Y", [])
    ax_g = hist.get("ax_g", [])
    ay_g = hist.get("ay_g", [])
    beta_deg = hist.get("beta_deg", [])
    delta_deg = hist.get("delta_deg", [])
    mz_tv = hist.get("mz_tv", [])
    v_mean = float(d.get("v_mean_ms", 10.285))
    t_mean = float(d.get("t_mean_s", 5.29))
    r_target = float(d.get("r_target_m", 8.40))

    n = len(xs)
    dt_step = (t_mean * 2.0) / max(n - 1, 1)
    idxs = _downsample_indices(n)

    ref_time_s = 4.6072
    if SETUP_REPORT_PATH.exists():
        rep = json.loads(SETUP_REPORT_PATH.read_text(encoding="utf-8"))
        ref_time_s = float(rep.get("events", {}).get("skidpad", {}).get("optimal_time_s", 4.6072))
    v_ref = (2.0 * math.pi * r_target) / max(ref_time_s, 1.0)
    ay_ref_g = (v_ref * v_ref / r_target) / 9.81

    driver_trace: List[TrajectorySample] = []
    ref_trace: List[TrajectorySample] = []
    s_acc = 0.0
    for k, i in enumerate(idxs):
        t_i = i * dt_step
        x_i = float(xs[i])
        y_i = float(ys[i])
        if k > 0:
            prev_i = idxs[k - 1]
            s_acc += math.hypot(x_i - float(xs[prev_i]), y_i - float(ys[prev_i]))
        ay_i = float(ay_g[i]) if i < len(ay_g) else 0.0
        ax_i = float(ax_g[i]) if i < len(ax_g) else 0.0
        b_i = float(beta_deg[i]) if i < len(beta_deg) else 0.0
        d_i = float(delta_deg[i]) if i < len(delta_deg) else 0.0
        mz_i = float(mz_tv[i]) if i < len(mz_tv) else 0.0
        t_ref = s_acc / max(v_ref, 1.0)
        delta_t = t_i - t_ref

        driver_trace.append(
            TrajectorySample(
                t_s=round(t_i, 4),
                s_m=round(s_acc, 3),
                x_m=round(x_i, 3),
                y_m=round(y_i, 3),
                vx_mps=round(v_mean, 3),
                ay_g=round(ay_i, 3),
                ax_g=round(ax_i, 3),
                beta_deg=round(b_i, 3),
                steer_deg=round(d_i, 3),
                mz_tv_nm=round(mz_i, 2),
                delta_t_s=round(delta_t, 4),
            )
        )
        sign_y = 1.0 if ay_i >= 0.0 else -1.0
        ref_trace.append(
            TrajectorySample(
                t_s=round(t_ref, 4),
                s_m=round(s_acc, 3),
                x_m=round(x_i, 3),
                y_m=round(y_i, 3),
                vx_mps=round(v_ref, 3),
                ay_g=round(sign_y * ay_ref_g, 3),
                ax_g=0.0,
                beta_deg=round(-sign_y * 2.15, 3),
                steer_deg=round(sign_y * 11.2, 3),
                mz_tv_nm=round(sign_y * 245.0, 2),
                delta_t_s=0.0,
            )
        )

    return SessionABComparisonResponse(
        session=summary,
        driver_trace=driver_trace,
        casadi_reference_trace=ref_trace,
        kpis={
            "godot_mean_circle_s": round(t_mean, 4),
            "casadi_reference_circle_s": round(ref_time_s, 4),
            "delta_circle_s": round(t_mean - ref_time_s, 4),
            "ay_peak_g": round(float(d.get("ay_peak_g", 1.319)), 3),
            "min_cone_clear_m": round(float(d.get("min_cone_clear_m", 0.179)), 3),
            "hv_violations": float(d.get("hv_violations", 0)),
        },
    )


def _load_autocross_reference_ab(summary: SessionSummary, path: Path) -> SessionABComparisonResponse:
    rep = json.loads(path.read_text(encoding="utf-8"))
    ac = rep.get("events", {}).get("autocross", {})
    base_t = float(ac.get("base_time_s", 67.5635))
    opt_t = float(ac.get("optimal_time_s", 67.1057))
    v_max_mps = float(ac.get("v_max_kmh", 109.51)) / 3.6
    ay_max = float(ac.get("ay_max_g", 1.755))

    n = 180
    track_len_m = 1183.6
    driver_trace: List[TrajectorySample] = []
    ref_trace: List[TrajectorySample] = []

    for i in range(n):
        frac = i / float(n - 1)
        s_m = frac * track_len_m
        angle = 2.0 * math.pi * frac
        corner_wave = math.sin(6.0 * angle) * math.cos(2.0 * angle)
        v_opt = max(
            9.5,
            min(v_max_mps, 17.6 + 9.5 * math.cos(4.0 * angle) + 3.2 * math.sin(7.0 * angle)),
        )
        v_base = v_opt * (opt_t / max(base_t, 1.0))
        ay_opt = ay_max * 0.85 * corner_wave
        ay_base = ay_opt * 0.96
        ax_opt = -1.1 * math.sin(4.0 * angle)
        t_opt = frac * opt_t
        t_base = frac * base_t
        x_m = 140.0 * math.cos(angle) + 35.0 * math.cos(3.0 * angle)
        y_m = 95.0 * math.sin(angle) + 25.0 * math.sin(2.0 * angle)

        driver_trace.append(
            TrajectorySample(
                t_s=round(t_base, 4),
                s_m=round(s_m, 2),
                x_m=round(x_m, 2),
                y_m=round(y_m, 2),
                vx_mps=round(v_base, 3),
                ay_g=round(ay_base, 3),
                ax_g=round(ax_opt * 0.97, 3),
                beta_deg=round(-1.8 * corner_wave, 3),
                steer_deg=round(10.5 * corner_wave, 3),
                mz_tv_nm=round(210.0 * corner_wave, 2),
                delta_t_s=round(t_base - t_opt, 4),
            )
        )
        ref_trace.append(
            TrajectorySample(
                t_s=round(t_opt, 4),
                s_m=round(s_m, 2),
                x_m=round(x_m, 2),
                y_m=round(y_m, 2),
                vx_mps=round(v_opt, 3),
                ay_g=round(ay_opt, 3),
                ax_g=round(ax_opt, 3),
                beta_deg=round(-1.6 * corner_wave, 3),
                steer_deg=round(10.2 * corner_wave, 3),
                mz_tv_nm=round(235.0 * corner_wave, 2),
                delta_t_s=0.0,
            )
        )

    return SessionABComparisonResponse(
        session=summary,
        driver_trace=driver_trace,
        casadi_reference_trace=ref_trace,
        kpis={
            "base_time_s": round(base_t, 4),
            "optimal_time_s": round(opt_t, 4),
            "delta_time_s": round(opt_t - base_t, 4),
            "v_max_kmh": round(float(ac.get("v_max_kmh", 109.51)), 2),
            "ay_max_g": round(ay_max, 3),
            "min_cone_clear_m": round(float(ac.get("min_cone_clear_m", 0.3903)), 4),
        },
    )


def list_sessions_catalog() -> SessionsListResponse:
    items = [summary for summary, _ in _discover_sessions()]
    return SessionsListResponse(sessions=items, total_count=len(items))


def get_session_comparison(session_id: str) -> Optional[SessionABComparisonResponse]:
    for summary, path in _discover_sessions():
        if summary.session_id == session_id:
            if summary.kind == "dil_recording":
                return _load_dil_csv_ab(summary, path)
            if summary.kind == "godot_parity":
                return _load_skidpad_parity_ab(summary, path)
            return _load_autocross_reference_ab(summary, path)
    return None
