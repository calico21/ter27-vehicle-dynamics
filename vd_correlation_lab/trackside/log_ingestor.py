#!/usr/bin/env python3
"""
Trackside Log Ingestor for TeR26 (RWD) and TeR27 (4WD).

Ingests:
  - .csv (standardized telemetry or raw CAN CSV)
  - .log (SocketCAN candump + TER.dbc)
  - .asc (Vector ASC + TER.dbc via python-can)
  - .blf (Vector BLF + TER.dbc via python-can)
  - .parquet (pre-processed ground truth)
  - dict / trace payloads (simulator traces)

Applies:
  - Mandatory RL <-> FR IR tire temperature sensor remap when tire temp
    channels are present (raw RL -> real FR, raw FR -> real RL).
  - ISO 8855 coordinate sign verification and enforcement (+X fwd, +Y left,
    +Z up, +wz left turn, +delta_rad left turn).
  - Uniform 50 Hz or 100 Hz resampling.
  - Maneuver segmentation: straight, steady_corner, heavy_braking, acceleration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from vd_correlation_lab.config.paths import DBC_PATH, VEHICLES_LAB_DIR
from vd_correlation_lab.control.sensors_ekf.log_parser import (
    CAN_SIGNAL_ALIASES,
    DbcDatabase,
    parse_candump_log,
)
from vd_correlation_lab.control.sensors_ekf.signal_conditioning import condition_telemetry

TIRE_TEMP_CAN_ALIASES: Dict[str, List[str]] = {
    "fl": ["T_tire_fl_c", "t_tire_fl_c", "TIRE_TEMP.flTemp", "flTemp", "T_FL", "tire_temp_fl"],
    "fr": ["T_tire_fr_c", "t_tire_fr_c", "TIRE_TEMP.frTemp", "frTemp", "T_FR", "tire_temp_fr"],
    "rl": ["T_tire_rl_c", "t_tire_rl_c", "TIRE_TEMP.rlTemp", "rlTemp", "T_RL", "tire_temp_rl"],
    "rr": ["T_tire_rr_c", "t_tire_rr_c", "TIRE_TEMP.rrTemp", "rrTemp", "T_RR", "tire_temp_rr"],
}


@dataclass
class IngestedRun:
    """Standardized 50/100 Hz telemetry container for trackside SysID and setup advice."""

    run_id: str
    source_path: str
    source_format: str
    vehicle_id: str
    fs_hz: float
    dt: float
    n_samples: int
    duration_s: float
    t_s: np.ndarray
    vx_ms: np.ndarray
    ax_ms2: np.ndarray
    ay_ms2: np.ndarray
    wz_rads: np.ndarray
    delta_rad: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    trq_limit_nm: np.ndarray
    wheel_rpm_fl: np.ndarray
    wheel_rpm_fr: np.ndarray
    wheel_rpm_rl: np.ndarray
    wheel_rpm_rr: np.ndarray
    t_tire_fl_c: np.ndarray
    t_tire_fr_c: np.ndarray
    t_tire_rl_c: np.ndarray
    t_tire_rr_c: np.ndarray
    is_active: np.ndarray
    segments: Dict[str, np.ndarray] = field(default_factory=dict)
    segment_stats: Dict[str, Any] = field(default_factory=dict)
    quality_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serialize run metadata, segment stats, and quality stats (optionally arrays)."""
        out: Dict[str, Any] = {
            "run_id": self.run_id,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "vehicle_id": self.vehicle_id,
            "fs_hz": float(self.fs_hz),
            "dt": float(self.dt),
            "n_samples": int(self.n_samples),
            "duration_s": float(self.duration_s),
            "segment_stats": dict(self.segment_stats),
            "quality_stats": dict(self.quality_stats),
        }
        if include_arrays:
            out["arrays"] = {
                "t_s": self.t_s.tolist(),
                "vx_ms": self.vx_ms.tolist(),
                "ax_ms2": self.ax_ms2.tolist(),
                "ay_ms2": self.ay_ms2.tolist(),
                "wz_rads": self.wz_rads.tolist(),
                "delta_rad": self.delta_rad.tolist(),
                "throttle": self.throttle.tolist(),
                "brake": self.brake.tolist(),
                "trq_limit_nm": self.trq_limit_nm.tolist(),
                "wheel_rpm_fl": self.wheel_rpm_fl.tolist(),
                "wheel_rpm_fr": self.wheel_rpm_fr.tolist(),
                "wheel_rpm_rl": self.wheel_rpm_rl.tolist(),
                "wheel_rpm_rr": self.wheel_rpm_rr.tolist(),
                "is_active": self.is_active.astype(bool).tolist(),
            }
        return out


def remap_tire_temperatures_rl_fr(
    t_fl: np.ndarray,
    t_fr: np.ndarray,
    t_rl: np.ndarray,
    t_rr: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Apply mandatory TeR26 IR tire temp wiring fix: raw RL <-> raw FR."""
    return {
        "fl": np.asarray(t_fl, dtype=np.float64).copy(),
        "fr": np.asarray(t_rl, dtype=np.float64).copy(),
        "rl": np.asarray(t_fr, dtype=np.float64).copy(),
        "rr": np.asarray(t_rr, dtype=np.float64).copy(),
    }


def segment_run(
    vx_ms: np.ndarray,
    ax_ms2: np.ndarray,
    ay_ms2: np.ndarray,
    wz_rads: np.ndarray,
    throttle: np.ndarray,
    brake: np.ndarray,
    dt: float,
    min_active_vx_ms: float = 3.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Segment a telemetry run into straights, steady-state corners, heavy braking, and accel."""
    n = len(vx_ms)
    active = vx_ms >= float(min_active_vx_ms)
    if n >= 5:
        ay_dot = np.gradient(ay_ms2, max(dt, 1e-4))
    else:
        ay_dot = np.zeros_like(ay_ms2)

    heavy_braking = active & ((ax_ms2 <= -3.5) | (brake >= 0.35))
    steady_corner = active & (~heavy_braking) & (np.abs(ay_ms2) >= 3.0) & (np.abs(ay_dot) < 12.0)
    straight = active & (~heavy_braking) & (np.abs(ay_ms2) < 1.5) & (np.abs(wz_rads) < 0.15)
    acceleration = active & (~heavy_braking) & (ax_ms2 >= 2.0) & (throttle >= 0.40)

    segments = {
        "active": active,
        "straight": straight,
        "steady_corner": steady_corner,
        "heavy_braking": heavy_braking,
        "acceleration": acceleration,
    }
    denom = max(n, 1)
    stats: Dict[str, Any] = {}
    for k, mask in segments.items():
        cnt = int(np.sum(mask))
        stats[k] = {
            "samples": cnt,
            "duration_s": round(cnt * dt, 3),
            "pct_total": round(100.0 * cnt / denom, 2),
        }
    return segments, stats


def _load_sensor_calib(vehicle_id: str) -> Dict[str, Any]:
    calib_path = VEHICLES_LAB_DIR / vehicle_id / "sensor_calibration.json"
    if calib_path.exists():
        return json.loads(calib_path.read_text(encoding="utf-8"))
    return {
        "vehicle_id": vehicle_id,
        "wheel_radius_m": 0.2032,
        "steer_ratio": 1.255,
        "filtering": {"fs_hz": 100.0, "butter_cutoff_hz": 15.0, "butter_order": 4, "min_active_vx_ms": 3.0},
    }


def _lowpass_if_needed(arr: np.ndarray, fs_hz: float, cutoff_hz: float = 15.0) -> np.ndarray:
    if len(arr) < 32 or cutoff_hz <= 0.0:
        return arr.copy()
    wn = min(cutoff_hz / (0.5 * fs_hz), 0.95)
    if wn <= 0.01:
        return arr.copy()
    sos = butter(4, wn, btype="low", output="sos")
    return sosfiltfilt(sos, arr)


def _decode_can_python_can(
    log_path: Path,
    dbc: DbcDatabase,
    fmt: str,
) -> Tuple[Dict[str, Tuple[List[float], List[float]]], float]:
    """Decode .asc or .blf CAN logs using python-can and TER.dbc."""
    import can  # type: ignore

    if fmt == "asc":
        reader = can.ASCReader(str(log_path))
    elif fmt == "blf":
        reader = can.BLFReader(str(log_path))
    else:
        raise ValueError(f"Unsupported python-can format: {fmt}")

    streams: Dict[str, Tuple[List[float], List[float]]] = {}
    t0: Optional[float] = None
    try:
        for msg in reader:
            ts = float(msg.timestamp)
            if t0 is None:
                t0 = ts
            t_rel = ts - t0
            can_id = int(msg.arbitration_id) & 0x1FFFFFFF
            decoded = dbc.decode_frame(can_id, bytes(msg.data))
            for k, v in decoded.items():
                if k not in streams:
                    streams[k] = ([], [])
                streams[k][0].append(t_rel)
                streams[k][1].append(float(v))
    finally:
        if hasattr(reader, "stop"):
            reader.stop()

    t_end = max((ts_list[-1] for ts_list, _ in streams.values() if ts_list), default=0.0)
    return streams, t_end


def _streams_to_conditioned_df(
    streams: Dict[str, Tuple[List[float], List[float]]],
    t_end: float,
    calib: Dict[str, Any],
    fs_hz: float,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, np.ndarray], bool]:
    """Resample decoded CAN streams onto uniform fs_hz grid and condition."""
    if t_end <= 0.05:
        raise ValueError("CAN log is empty or contains no decodable frames.")

    dt = 1.0 / fs_hz
    t_grid = np.arange(0.0, t_end, dt, dtype=np.float64)
    df = pd.DataFrame({"t_s": t_grid})
    present_counts: Dict[str, int] = {}

    for col, aliases in CAN_SIGNAL_ALIASES.items():
        st = None
        for alias in aliases:
            if alias in streams and len(streams[alias][0]) >= 2:
                st = streams[alias]
                break
        if st is None:
            df[col] = np.nan
            present_counts[col] = 0
        else:
            df[col] = np.interp(t_grid, np.asarray(st[0], dtype=np.float64), np.asarray(st[1], dtype=np.float64))
            present_counts[col] = len(st[0])

    # Extract optional tire temperature streams
    tire_temps: Dict[str, np.ndarray] = {}
    temps_present = False
    for corner, aliases in TIRE_TEMP_CAN_ALIASES.items():
        st = None
        for alias in aliases:
            if alias in streams and len(streams[alias][0]) >= 2:
                st = streams[alias]
                break
        if st is not None:
            temps_present = True
            tire_temps[corner] = np.interp(
                t_grid, np.asarray(st[0], dtype=np.float64), np.asarray(st[1], dtype=np.float64)
            )
        else:
            tire_temps[corner] = np.full_like(t_grid, np.nan, dtype=np.float64)

    calib_local = json.loads(json.dumps(calib))
    calib_local.setdefault("filtering", {})["fs_hz"] = float(fs_hz)
    df_cond, meta = condition_telemetry(df, streams, present_counts, calib_local, is_preconditioned_csv=False)

    # Convert steer_deg -> delta_rad (road wheel angle)
    if "delta_rad" not in df_cond.columns:
        steer_deg = df_cond["steer_deg"].to_numpy(dtype=np.float64) if "steer_deg" in df_cond.columns else np.zeros_like(t_grid)
        df_cond["delta_rad"] = np.radians(steer_deg)

    total_expected = max(len(CAN_SIGNAL_ALIASES), 1)
    missing_cols = sum(1 for v in present_counts.values() if v == 0)
    meta["channel_missing_rate"] = round(float(missing_cols) / float(total_expected), 4)
    return df_cond, meta, tire_temps, temps_present


def _enforce_iso8855_signs(
    vx: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    wz: np.ndarray,
    delta_rad: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Verify and enforce ISO 8855 sign consistency (+wz left, +ay left, +delta left)."""
    m = (vx > 3.0) & np.isfinite(ay) & np.isfinite(wz) & np.isfinite(delta_rad)
    ay_flipped = False
    delta_flipped = False

    corr_wz_ay = 0.0
    corr_delta_wz = 0.0

    if int(np.sum(m)) > 20 and float(np.std(wz[m])) > 1e-3:
        if float(np.std(ay[m])) > 0.1:
            corr_wz_ay = float(np.corrcoef(wz[m], ay[m])[0, 1])
            if corr_wz_ay < -0.30:
                ay = -ay
                corr_wz_ay = -corr_wz_ay
                ay_flipped = True
        if float(np.std(delta_rad[m])) > 1e-4:
            corr_delta_wz = float(np.corrcoef(delta_rad[m], wz[m])[0, 1])
            if corr_delta_wz < -0.30:
                delta_rad = -delta_rad
                corr_delta_wz = -corr_delta_wz
                delta_flipped = True

    audit = {
        "iso8855_corr_wz_ay": round(corr_wz_ay, 4),
        "iso8855_corr_delta_wz": round(corr_delta_wz, 4),
        "iso8855_ay_flipped": ay_flipped,
        "iso8855_delta_flipped": delta_flipped,
        "iso8855_passed": bool((corr_wz_ay >= -0.1) and (corr_delta_wz >= -0.1)),
    }
    return ay, wz, delta_rad, audit


def _resample_arrays_uniform(
    t_in: np.ndarray,
    arrays: Dict[str, np.ndarray],
    fs_hz: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Resample arrays onto uniform t_grid at fs_hz."""
    t0 = float(t_in[0])
    t1 = float(t_in[-1])
    dt = 1.0 / fs_hz
    t_rel = t_in - t0
    t_grid = np.arange(0.0, max(t1 - t0, dt), dt, dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    for k, arr in arrays.items():
        a = np.asarray(arr, dtype=np.float64)
        if np.all(np.isnan(a)):
            out[k] = np.full_like(t_grid, np.nan, dtype=np.float64)
        else:
            valid = np.isfinite(a)
            if int(np.sum(valid)) >= 2:
                out[k] = np.interp(t_grid, t_rel[valid], a[valid])
            else:
                out[k] = np.full_like(t_grid, float(a[valid][0]) if np.any(valid) else 0.0, dtype=np.float64)
    return t_grid, out


def _finalize_ingested_run(
    run_id: str,
    source_path: str,
    source_format: str,
    vehicle_id: str,
    fs_hz: float,
    t_s: np.ndarray,
    arrays: Dict[str, np.ndarray],
    tire_temps: Dict[str, np.ndarray],
    temps_present: bool,
    apply_tire_temp_remap: bool,
    meta: Dict[str, Any],
) -> IngestedRun:
    dt = 1.0 / float(fs_hz)

    if temps_present and apply_tire_temp_remap:
        remapped = remap_tire_temperatures_rl_fr(
            tire_temps["fl"], tire_temps["fr"], tire_temps["rl"], tire_temps["rr"]
        )
        t_fl, t_fr, t_rl, t_rr = remapped["fl"], remapped["fr"], remapped["rl"], remapped["rr"]
        remap_applied = True
    else:
        t_fl = tire_temps["fl"]
        t_fr = tire_temps["fr"]
        t_rl = tire_temps["rl"]
        t_rr = tire_temps["rr"]
        remap_applied = False

    vx = np.nan_to_num(arrays["vx_ms"], nan=0.0)
    ax = np.nan_to_num(arrays["ax_ms2"], nan=0.0)
    ay = np.nan_to_num(arrays["ay_ms2"], nan=0.0)
    wz = np.nan_to_num(arrays["wz_rads"], nan=0.0)
    delta_rad = np.nan_to_num(arrays["delta_rad"], nan=0.0)
    throttle = np.clip(np.nan_to_num(arrays["throttle"], nan=0.0), 0.0, 1.0)
    brake = np.clip(np.nan_to_num(arrays["brake"], nan=0.0), 0.0, 1.0)
    trq_limit = np.nan_to_num(arrays["trq_limit_nm"], nan=120.0)
    rpm_fl = np.nan_to_num(arrays["wheel_rpm_fl"], nan=0.0)
    rpm_fr = np.nan_to_num(arrays["wheel_rpm_fr"], nan=0.0)
    rpm_rl = np.nan_to_num(arrays["wheel_rpm_rl"], nan=0.0)
    rpm_rr = np.nan_to_num(arrays["wheel_rpm_rr"], nan=0.0)

    ay, wz, delta_rad, iso_audit = _enforce_iso8855_signs(vx, ax, ay, wz, delta_rad)
    segments, seg_stats = segment_run(vx, ax, ay, wz, throttle, brake, dt)
    active = segments["active"]

    vx_p95 = float(np.percentile(vx, 95)) if len(vx) > 0 else 0.0
    ay_p95 = float(np.percentile(np.abs(ay), 95)) if len(ay) > 0 else 0.0
    is_track = (vx_p95 >= 8.5) and (ay_p95 >= 4.5)

    quality_stats: Dict[str, Any] = {
        "log_type": meta.get("log_type", "track" if is_track else "paddock_slow"),
        "vx_p95_ms": round(vx_p95, 3),
        "ay_p95_ms2": round(ay_p95, 3),
        "ax_min_ms2": round(float(np.min(ax)) if len(ax) else 0.0, 3),
        "ax_max_ms2": round(float(np.max(ax)) if len(ax) else 0.0, 3),
        "active_samples": int(np.sum(active)),
        "total_samples": int(len(t_s)),
        "emi_spikes_filtered": int(meta.get("emi_spikes_filtered", 0)),
        "channel_missing_rate": float(meta.get("channel_missing_rate", 0.0)),
        "tire_temps_present": bool(temps_present),
        "tire_temp_remap_applied": bool(remap_applied),
        "tire_temp_remap_rule": "raw_RL <-> raw_FR (FL, RR unchanged)" if remap_applied else "none",
        **iso_audit,
    }

    return IngestedRun(
        run_id=run_id,
        source_path=source_path,
        source_format=source_format,
        vehicle_id=vehicle_id,
        fs_hz=float(fs_hz),
        dt=float(dt),
        n_samples=int(len(t_s)),
        duration_s=round(float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0, 4),
        t_s=t_s,
        vx_ms=vx,
        ax_ms2=ax,
        ay_ms2=ay,
        wz_rads=wz,
        delta_rad=delta_rad,
        throttle=throttle,
        brake=brake,
        trq_limit_nm=trq_limit,
        wheel_rpm_fl=rpm_fl,
        wheel_rpm_fr=rpm_fr,
        wheel_rpm_rl=rpm_rl,
        wheel_rpm_rr=rpm_rr,
        t_tire_fl_c=t_fl,
        t_tire_fr_c=t_fr,
        t_tire_rl_c=t_rl,
        t_tire_rr_c=t_rr,
        is_active=active,
        segments=segments,
        segment_stats=seg_stats,
        quality_stats=quality_stats,
    )


def ingest_trace(
    trace: Dict[str, Any],
    vehicle_id: str = "ter27_4wd",
    fs_hz: float = 100.0,
    run_id: str = "trace_run",
    apply_tire_temp_remap: Optional[bool] = None,
) -> IngestedRun:
    """Ingest an in-memory dictionary / simulator trace into a standardized IngestedRun."""
    if fs_hz not in (50.0, 100.0):
        raise ValueError(f"fs_hz must be 50.0 or 100.0, got {fs_hz}")
    if apply_tire_temp_remap is None:
        apply_tire_temp_remap = (vehicle_id == "ter26_rwd")

    t_in = np.asarray(trace["t_s"], dtype=np.float64)
    n_in = len(t_in)
    zeros = np.zeros(n_in, dtype=np.float64)

    if "delta_rad" in trace:
        delta_in = np.asarray(trace["delta_rad"], dtype=np.float64)
    elif "steer_deg" in trace:
        delta_in = np.radians(np.asarray(trace["steer_deg"], dtype=np.float64))
    else:
        delta_in = zeros

    raw_arrays: Dict[str, np.ndarray] = {
        "vx_ms": np.asarray(trace.get("vx_ms", zeros), dtype=np.float64),
        "ax_ms2": np.asarray(trace.get("ax_ms2", zeros), dtype=np.float64),
        "ay_ms2": np.asarray(trace.get("ay_ms2", zeros), dtype=np.float64),
        "wz_rads": np.asarray(trace.get("wz_rads", zeros), dtype=np.float64),
        "delta_rad": delta_in,
        "throttle": np.asarray(trace.get("throttle", zeros), dtype=np.float64),
        "brake": np.asarray(trace.get("brake", zeros), dtype=np.float64),
        "trq_limit_nm": np.asarray(trace.get("trq_limit_nm", np.full(n_in, 120.0)), dtype=np.float64),
        "wheel_rpm_fl": np.asarray(trace.get("wheel_rpm_fl", zeros), dtype=np.float64),
        "wheel_rpm_fr": np.asarray(trace.get("wheel_rpm_fr", zeros), dtype=np.float64),
        "wheel_rpm_rl": np.asarray(trace.get("wheel_rpm_rl", zeros), dtype=np.float64),
        "wheel_rpm_rr": np.asarray(trace.get("wheel_rpm_rr", zeros), dtype=np.float64),
    }

    temps_present = False
    for corner, aliases in TIRE_TEMP_CAN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in trace:
                found = np.asarray(trace[alias], dtype=np.float64)
                temps_present = True
                break
        raw_arrays[f"_t_{corner}"] = found if found is not None else np.full(n_in, np.nan, dtype=np.float64)

    t_grid, resampled = _resample_arrays_uniform(t_in, raw_arrays, fs_hz=fs_hz)
    tire_temps = {c: resampled.pop(f"_t_{c}") for c in ("fl", "fr", "rl", "rr")}
    return _finalize_ingested_run(
        run_id=run_id,
        source_path="in_memory_trace",
        source_format="trace_dict",
        vehicle_id=vehicle_id,
        fs_hz=fs_hz,
        t_s=t_grid,
        arrays=resampled,
        tire_temps=tire_temps,
        temps_present=temps_present,
        apply_tire_temp_remap=bool(apply_tire_temp_remap),
        meta={},
    )


def ingest_log(
    log_path: Union[str, Path],
    vehicle_id: str = "ter26_rwd",
    fs_hz: float = 100.0,
    dbc_path: Optional[Union[str, Path]] = None,
    apply_tire_temp_remap: Optional[bool] = None,
    run_id: Optional[str] = None,
) -> IngestedRun:
    """Ingest .csv, .log, .asc, .blf, or .parquet telemetry into a standardized IngestedRun."""
    if fs_hz not in (50.0, 100.0):
        raise ValueError(f"fs_hz must be 50.0 or 100.0, got {fs_hz}")

    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry log not found: {path}")

    if apply_tire_temp_remap is None:
        apply_tire_temp_remap = (vehicle_id == "ter26_rwd")

    ext = path.suffix.lower()
    rid = run_id or path.stem
    calib = _load_sensor_calib(vehicle_id)
    dbc_file = Path(dbc_path) if dbc_path is not None else DBC_PATH

    if ext in (".log", ".asc", ".blf"):
        dbc = DbcDatabase(dbc_file)
        if ext == ".log":
            streams, t_end = parse_candump_log(path, dbc)
            fmt = "candump_log"
        elif ext == ".asc":
            streams, t_end = _decode_can_python_can(path, dbc, "asc")
            fmt = "asc"
        else:
            streams, t_end = _decode_can_python_can(path, dbc, "blf")
            fmt = "blf"

        df_cond, meta, tire_temps, temps_present = _streams_to_conditioned_df(
            streams, t_end, calib=calib, fs_hz=fs_hz
        )
        t_s = df_cond["t_s"].to_numpy(dtype=np.float64)
        arrays = {
            "vx_ms": df_cond["vx_ms"].to_numpy(dtype=np.float64),
            "ax_ms2": df_cond["ax_ms2"].to_numpy(dtype=np.float64),
            "ay_ms2": df_cond["ay_ms2"].to_numpy(dtype=np.float64),
            "wz_rads": df_cond["wz_rads"].to_numpy(dtype=np.float64),
            "delta_rad": df_cond["delta_rad"].to_numpy(dtype=np.float64),
            "throttle": df_cond["throttle"].to_numpy(dtype=np.float64),
            "brake": df_cond["brake"].to_numpy(dtype=np.float64),
            "trq_limit_nm": df_cond["trq_limit_nm"].to_numpy(dtype=np.float64),
            "wheel_rpm_fl": df_cond["wheel_rpm_fl"].to_numpy(dtype=np.float64),
            "wheel_rpm_fr": df_cond["wheel_rpm_fr"].to_numpy(dtype=np.float64),
            "wheel_rpm_rl": df_cond["wheel_rpm_rl"].to_numpy(dtype=np.float64),
            "wheel_rpm_rr": df_cond["wheel_rpm_rr"].to_numpy(dtype=np.float64),
        }
        return _finalize_ingested_run(
            run_id=rid,
            source_path=str(path),
            source_format=fmt,
            vehicle_id=vehicle_id,
            fs_hz=fs_hz,
            t_s=t_s,
            arrays=arrays,
            tire_temps=tire_temps,
            temps_present=temps_present,
            apply_tire_temp_remap=bool(apply_tire_temp_remap),
            meta=meta,
        )

    if ext in (".csv", ".parquet"):
        if ext == ".csv":
            header_cols = pd.read_csv(path, nrows=0).columns.tolist()
            if "Time" in header_cols and "ID" in header_cols:
                from vd_correlation_lab.validation.differentiable_ter26_correlation import (
                    _load_and_condition_can_csv,
                )

                dt_target = 1.0 / float(fs_hz)
                cond = _load_and_condition_can_csv(path, calib=calib, dt=dt_target)
                t_s = cond["t_s"]
                n_s = len(t_s)
                zeros = np.zeros(n_s, dtype=np.float64)
                arrays = {
                    "vx_ms": cond["vx_ms"],
                    "ax_ms2": cond["ax_ms2"],
                    "ay_ms2": cond["ay_ms2"],
                    "wz_rads": cond["wz_rads"],
                    "delta_rad": cond["delta_rad"],
                    "throttle": cond["throttle"],
                    "brake": cond["brake"],
                    "trq_limit_nm": cond.get("trq_limit_nm", np.full(n_s, 120.0)),
                    "wheel_rpm_fl": cond.get("wheel_rpm_fl", zeros),
                    "wheel_rpm_fr": cond.get("wheel_rpm_fr", zeros),
                    "wheel_rpm_rl": cond["wheel_rpm_rl"],
                    "wheel_rpm_rr": cond["wheel_rpm_rr"],
                }
                tire_temps = {
                    c: np.full(n_s, np.nan, dtype=np.float64) for c in ("fl", "fr", "rl", "rr")
                }
                return _finalize_ingested_run(
                    run_id=rid,
                    source_path=str(path),
                    source_format="can_csv",
                    vehicle_id=vehicle_id,
                    fs_hz=fs_hz,
                    t_s=t_s,
                    arrays=arrays,
                    tire_temps=tire_temps,
                    temps_present=False,
                    apply_tire_temp_remap=bool(apply_tire_temp_remap),
                    meta=cond.get("calibration_audit", {}),
                )

        df = pd.read_csv(path) if ext == ".csv" else pd.read_parquet(path)
        fmt = "csv" if ext == ".csv" else "parquet"
        t_col = "t_s" if "t_s" in df.columns else df.columns[0]
        t_in = df[t_col].to_numpy(dtype=np.float64)
        n_in = len(t_in)
        zeros = np.zeros(n_in, dtype=np.float64)

        def _pick(cols: List[str], default: np.ndarray) -> np.ndarray:
            for c in cols:
                if c in df.columns:
                    return df[c].to_numpy(dtype=np.float64)
            return default

        delta_in = _pick(["delta_rad", "delta_wheel_rad"], np.full(n_in, np.nan))
        if np.all(np.isnan(delta_in)):
            steer_deg = _pick(["steer_deg", "STEER.ANGLE"], zeros)
            delta_in = np.radians(steer_deg)

        raw_arrays = {
            "vx_ms": _pick(["vx_ms", "vx_ekf_ms"], zeros),
            "ax_ms2": _pick(["ax_ms2", "ax_cg_ms2"], zeros),
            "ay_ms2": _pick(["ay_ms2", "ay_cg_ms2"], zeros),
            "wz_rads": _pick(["wz_rads"], zeros),
            "delta_rad": delta_in,
            "throttle": _pick(["throttle"], zeros),
            "brake": _pick(["brake"], zeros),
            "trq_limit_nm": _pick(["trq_limit_nm"], np.full(n_in, 120.0)),
            "wheel_rpm_fl": _pick(["wheel_rpm_fl"], zeros),
            "wheel_rpm_fr": _pick(["wheel_rpm_fr"], zeros),
            "wheel_rpm_rl": _pick(["wheel_rpm_rl"], zeros),
            "wheel_rpm_rr": _pick(["wheel_rpm_rr"], zeros),
        }

        temps_present = False
        for corner, aliases in TIRE_TEMP_CAN_ALIASES.items():
            found = None
            for alias in aliases:
                if alias in df.columns and df[alias].notna().sum() > 0:
                    found = df[alias].to_numpy(dtype=np.float64)
                    temps_present = True
                    break
            raw_arrays[f"_t_{corner}"] = found if found is not None else np.full(n_in, np.nan, dtype=np.float64)

        t_grid, resampled = _resample_arrays_uniform(t_in, raw_arrays, fs_hz=fs_hz)
        for sig in ("ax_ms2", "ay_ms2", "wz_rads"):
            resampled[sig] = _lowpass_if_needed(resampled[sig], fs_hz=fs_hz, cutoff_hz=15.0)
        tire_temps = {c: resampled.pop(f"_t_{c}") for c in ("fl", "fr", "rl", "rr")}

        return _finalize_ingested_run(
            run_id=rid,
            source_path=str(path),
            source_format=fmt,
            vehicle_id=vehicle_id,
            fs_hz=fs_hz,
            t_s=t_grid,
            arrays=resampled,
            tire_temps=tire_temps,
            temps_present=temps_present,
            apply_tire_temp_remap=bool(apply_tire_temp_remap),
            meta={},
        )

    raise ValueError(f"Unsupported telemetry format '{ext}' for {path}")
