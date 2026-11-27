#!/usr/bin/env python3
"""
Pipeline de acondicionamiento de señal parametrizado por sensor_calibration.json.
Aplica calibración IMU (rotación, retardo, brazo de palanca, escala),
limpieza EMI de BPPS, offset estático y adelanto CAN de dirección,
TRQ_LIMIT(t) por retenedor de orden cero, corte Brake-Over-Throttle,
filtrado Butterworth fase cero (conservando señal cruda para FFT)
y clasificación pista vs paddock_slow.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, filtfilt

G_TO_MS2 = 9.80665


def shift_series(arr: np.ndarray, shift_samples: int) -> np.ndarray:
    """Desplaza una serie temporal en muestras (positivo = retrasar, negativo = adelantar)."""
    if shift_samples == 0 or len(arr) < 2:
        return arr.copy()
    out = np.empty_like(arr)
    if shift_samples > 0:
        k = min(shift_samples, len(arr) - 1)
        out[k:] = arr[:-k]
        out[:k] = arr[0]
    else:
        k = min(-shift_samples, len(arr) - 1)
        out[:-k] = arr[k:]
        out[-k:] = arr[-1]
    return out


def zero_phase_lowpass(arr: np.ndarray, dt: float, cutoff_hz: float = 15.0, order: int = 4) -> np.ndarray:
    """Filtro Butterworth fase cero (filtfilt) para evitar distorsión de fase."""
    if len(arr) < 20 or cutoff_hz <= 0.0:
        return arr.copy()
    fs = 1.0 / max(dt, 1e-4)
    wn = min(cutoff_hz / (0.5 * fs), 0.95)
    if wn <= 0.01:
        return arr.copy()
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, arr)


def condition_telemetry(
    df: pd.DataFrame,
    raw_streams: Dict[str, Tuple[List[float], List[float]]],
    present_counts: Dict[str, int],
    calib: Dict[str, Any],
    is_preconditioned_csv: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aplica el pipeline completo de acondicionamiento y devuelve (df_acondicionado, metadata)."""
    filt_cfg = calib.get("filtering", {})
    fs_hz = float(filt_cfg.get("fs_hz", 100.0))
    dt = 1.0 / fs_hz
    cutoff_hz = float(filt_cfg.get("butter_cutoff_hz", 15.0))
    butter_order = int(filt_cfg.get("butter_order", 4))
    min_active_vx = float(filt_cfg.get("min_active_vx_ms", 3.0))

    if is_preconditioned_csv:
        out = df.copy()
        for c in ("ax_ms2", "ay_ms2", "wz_rads"):
            if c in out.columns:
                out[c.replace("_ms2", "_raw_ms2").replace("_rads", "_raw_rads")] = out[c].to_numpy()
                out[c] = zero_phase_lowpass(out[c].to_numpy(), dt, cutoff_hz, butter_order)
        for w in ("fl", "fr", "rl", "rr"):
            col = f"wheel_rpm_{w}"
            if col in out.columns:
                out[f"{col}_raw"] = out[col].to_numpy()
        vx_arr = out["vx_ms"].to_numpy() if "vx_ms" in out.columns else np.zeros(len(out))
        ay_arr = out["ay_ms2"].to_numpy() if "ay_ms2" in out.columns else np.zeros(len(out))
        vx_p95 = float(np.percentile(vx_arr, 95)) if len(vx_arr) > 0 else 0.0
        ay_p95 = float(np.percentile(np.abs(ay_arr), 95)) if len(ay_arr) > 0 else 0.0
        is_track = (
            vx_p95 >= float(filt_cfg.get("paddock_vx_p95_thresh_ms", 8.5))
            and ay_p95 >= float(filt_cfg.get("paddock_ay_p95_thresh_ms2", 4.5))
        )
        active = vx_arr >= min_active_vx
        out["is_active"] = active
        meta = {
            "log_type": "track" if is_track else "paddock_slow",
            "vx_p95_ms": vx_p95,
            "ay_p95_ms2": ay_p95,
            "emi_spikes_filtered": 0,
            "steer_zero_deg": 0.0,
            "imu_mode": "synthetic_clean",
            "active_samples": int(np.sum(active)),
            "total_samples": len(out),
        }
        return out, meta

    t_grid = df["t_s"].to_numpy()
    wheel_r = float(calib.get("wheel_radius_m", 0.2032))
    steer_ratio = float(calib.get("steer_ratio", 1.255))
    imu_cfg = calib.get("imu", {})
    brk_cfg = calib.get("brakes", {})
    str_cfg = calib.get("steering", {})
    pwr_cfg = calib.get("powertrain", {})

    # 1. vel longitudinal de referencia
    rear_ok = present_counts.get("wheel_rpm_rl", 0) > 0 and present_counts.get("wheel_rpm_rr", 0) > 0
    front_v_ok = present_counts.get("front_vx_av_ms", 0) > 0 and float(np.nanstd(df["front_vx_av_ms"])) > 0.05
    if front_v_ok:
        vx_raw = df["front_vx_av_ms"].to_numpy()
        vx_src = "Front_V.vx_av"
    elif rear_ok:
        vx_raw = np.maximum(0.5 * (df["wheel_rpm_rl"].to_numpy() + df["wheel_rpm_rr"].to_numpy()), 0.0) * (2.0 * math.pi / 60.0) * wheel_r
        vx_src = "WheelInfo_rear_RPM_avg"
    elif present_counts.get("wheel_speed_kmh", 0) > 0:
        vx_raw = df["wheel_speed_kmh"].to_numpy() / 3.6
        vx_src = "WheelInfo.speed"
    else:
        vx_raw = np.zeros_like(t_grid)
        vx_src = "none"
    df["vx_ms"] = np.nan_to_num(vx_raw, nan=0.0)
    for w in ("fl", "fr", "rl", "rr"):
        col = f"wheel_rpm_{w}"
        df[col] = np.nan_to_num(df[col].to_numpy(), nan=0.0)
        df[f"{col}_raw"] = df[col].to_numpy()

    rest_mask = df["vx_ms"].to_numpy() < 0.35
    if int(np.sum(rest_mask)) < 20:
        rest_mask = df["vx_ms"].to_numpy() < (np.percentile(df["vx_ms"], 10) + 0.5)

    # 2. guiñada imu (0x017 / 0x019)
    wz_raw = np.radians(df["imu_wz_degs"].fillna(0.0).to_numpy()) * float(imu_cfg.get("scale_wz", 1.0))
    if np.any(rest_mask) and present_counts.get("imu_wz_degs", 0) > 20:
        wz_raw = wz_raw - float(np.mean(wz_raw[rest_mask]))
    df["wz_raw_rads"] = wz_raw
    df["wz_rads"] = zero_phase_lowpass(wz_raw, dt, cutoff_hz, butter_order)

    # 3. pedales: apps zona muerta + bpps filtro emi + escala 17.5 bar + brake-over-throttle
    apps_max = float(np.nanmax(df["apps_av_raw"])) if present_counts.get("apps_av_raw", 0) > 0 else 0.0
    apps_denom = 255.0 if apps_max > 102.0 or "APPS.APPS_AV" in raw_streams else 100.0
    raw_thr = (df["apps_av_raw"] / apps_denom).clip(0.0, 1.0).fillna(0.0).to_numpy()
    vx_dot_s = uniform_filter1d(np.gradient(uniform_filter1d(df["vx_ms"].to_numpy(), 15), dt), 15)

    bpps_full_bar = float(brk_cfg.get("bpps_full_bar", 17.5))
    emi_apps_th = float(brk_cfg.get("bpps_emi_apps_thresh", 0.25))
    emi_count = 0
    if present_counts.get("bpps_bar", 0) > 0:
        bp = np.nan_to_num(df["bpps_bar"].to_numpy(), nan=0.0)
        emi_mask = (
            (bp > 25.0)
            | ((bp > 2.0) & (raw_thr > emi_apps_th) & (vx_dot_s > 0.15))
            | ((bp > 4.0) & (raw_thr > 0.08) & (vx_dot_s > -0.40))
        )
        emi_count = int(np.sum(emi_mask))
        bp_clean = median_filter(np.where(emi_mask, 0.0, bp), size=3)
        bp0 = min(max(float(np.nanpercentile(bp_clean, 5)), 0.0), 0.5)
        brk_raw = np.clip((bp_clean - bp0) / max(bpps_full_bar - bp0, 5.0), 0.0, 1.0)
    else:
        bp_clean = np.zeros_like(t_grid)
        brk_raw = np.zeros_like(t_grid)

    apps_dz = float(pwr_cfg.get("apps_deadzone", 0.075))
    thr_raw = np.clip((raw_thr - apps_dz) / max(1.0 - apps_dz, 1e-3), 0.0, 1.0)
    bot_thresh = float(brk_cfg.get("brake_over_throttle_thresh", 0.05))
    thr_raw = np.where(brk_raw > bot_thresh, 0.0, thr_raw)
    df["throttle"] = thr_raw
    df["brake"] = brk_raw
    df["bpps_bar_clean"] = bp_clean

    # 4. dirección can (+50 ms) y cero mecánico x tanda
    steer_raw_w = (df["steer_raw_deg"] / max(steer_ratio, 1e-3)).fillna(0.0).to_numpy()
    steer_shift = int(str_cfg.get("can_shift_samples", 5))
    steer_aligned = shift_series(steer_raw_w, steer_shift)
    vx_zero_min = float(str_cfg.get("auto_zero_vx_min_ms", 4.0))
    m_str = (df["vx_ms"].to_numpy() > vx_zero_min) & (np.abs(wz_raw) < 0.25) & np.isfinite(steer_aligned) & np.isfinite(wz_raw)
    if int(np.sum(m_str)) > 50:
        curv = wz_raw[m_str] / df["vx_ms"].to_numpy()[m_str]
        if float(np.std(curv)) > 1e-5:
            steer_zero_deg = float(np.clip(np.polyfit(curv, steer_aligned[m_str], 1)[1], -4.0, 4.0))
        else:
            steer_zero_deg = float(np.clip(np.median(steer_aligned[m_str]), -4.0, 4.0))
    elif np.any(rest_mask):
        steer_zero_deg = float(np.clip(np.median(steer_aligned[rest_mask]), -4.0, 4.0))
    else:
        steer_zero_deg = 0.0
    df["steer_deg"] = steer_aligned - steer_zero_deg

    # 5. trq_limit(t) x retenedor de orden cero
    def_trq = float(pwr_cfg.get("default_trq_limit_nm", 120.0))
    t_trq_st = raw_streams.get("ECU_CONFIG.TRQ_LIMIT") or raw_streams.get("COMMAND.trqLimit")
    if t_trq_st is not None and len(t_trq_st[0]) >= 1:
        t_arr = np.asarray(t_trq_st[0], dtype=np.float64)
        v_arr = np.asarray(t_trq_st[1], dtype=np.float64)
        valid_trq = v_arr > 10.0
        if np.any(valid_trq):
            t_arr, v_arr = t_arr[valid_trq], v_arr[valid_trq]
            idx_zoh = np.clip(np.searchsorted(t_arr, t_grid, side="right") - 1, 0, len(v_arr) - 1)
            df["trq_limit_nm"] = v_arr[idx_zoh]
        else:
            df["trq_limit_nm"] = def_trq
    else:
        df["trq_limit_nm"] = def_trq

    # 6. calibración física imu (0x01a accel)
    accel_present = present_counts.get("imu_az_raw", 0) > 0
    accel_scale = G_TO_MS2 if (accel_present and 0.45 < float(np.nanmedian(np.abs(df["imu_az_raw"]))) < 1.65) else 1.0
    imu_mode = "identity"
    if accel_present:
        ax_0 = np.nan_to_num(df["imu_ax_raw"].to_numpy() * accel_scale, nan=0.0)
        ay_0 = np.nan_to_num(df["imu_ay_raw"].to_numpy() * accel_scale, nan=0.0)
        if np.any(rest_mask):
            ax_0 -= float(np.mean(ax_0[rest_mask]))
            ay_0 -= float(np.mean(ay_0[rest_mask]))

        adv_samples = int(imu_cfg.get("accel_advance_samples", 11))
        ax_0 = shift_series(ax_0, -adv_samples)
        ay_0 = shift_series(ay_0, -adv_samples)

        vx_arr = df["vx_ms"].to_numpy()
        vx_s = uniform_filter1d(vx_arr, 11)
        wz_s = uniform_filter1d(wz_raw, 9)
        dvx_dt = np.gradient(vx_s, dt)
        cent_ay = vx_s * wz_s
        win = np.ones(21) / 21.0
        dvx_c = np.convolve(dvx_dt, win, mode="same")
        cent_c = np.convolve(cent_ay, win, mode="same")
        ax_c = np.convolve(ax_0, win, mode="same")
        ay_c = np.convolve(ay_0, win, mode="same")
        dyn = (vx_arr > 2.0) & (np.abs(cent_c) + np.abs(dvx_c) > 0.25)

        yaw_deg = float(imu_cfg.get("yaw_offset_deg", -22.25))
        x_lever = float(imu_cfg.get("x_lever_m", 0.78))

        if int(np.sum(dyn)) > 80 and np.std(dvx_c[dyn]) > 0.1:
            c_xx = float(np.corrcoef(ax_c[dyn], dvx_c[dyn])[0, 1])
            c_yx = float(np.corrcoef(ay_c[dyn], dvx_c[dyn])[0, 1])
            lat_ok = float(np.std(cent_c[dyn])) > 0.55
            c_xy = float(np.corrcoef(ax_c[dyn], cent_c[dyn])[0, 1]) if lat_ok else 0.0
            c_yy = float(np.corrcoef(ay_c[dyn], cent_c[dyn])[0, 1]) if lat_ok else 0.0
            use_rot90 = ((abs(c_xy) + abs(c_yx)) > (abs(c_xx) + abs(c_yy) + 0.15)) if lat_ok else (abs(c_yx) > abs(c_xx) + 0.12)
            if use_rot90:
                sx = 1.0 if c_yx >= 0.0 else -1.0
                sy = (1.0 if c_xy >= 0.0 else -1.0) if (lat_ok and abs(c_xy) > 0.3) else -sx
                ax_sw = sx * ay_0
                ay_sw = sy * ax_0
            else:
                sx = 1.0 if c_xx >= 0.0 else -1.0
                sy = 1.0 if c_yy >= 0.0 else -1.0
                ax_sw = sx * ax_0
                ay_sw = sy * ay_0

            th_z = math.radians(yaw_deg)
            ax_rot = ax_sw * math.cos(th_z) - ay_sw * math.sin(th_z)
            ay_rot = ax_sw * math.sin(th_z) + ay_sw * math.cos(th_z)
            ax_cal = (ax_rot + x_lever * (wz_s ** 2)) * float(imu_cfg.get("scale_ax", 1.0))

            if np.sum(vx_arr > 2.0) > 50:
                m_mv = vx_arr > 2.0
                pitch_bias = float(np.mean(ax_cal[m_mv]) - np.mean(dvx_dt[m_mv]))
                ax_cal = np.where(vx_arr > 0.35, ax_cal - pitch_bias, ax_cal)

            if lat_ok:
                slope_y = float(np.clip(np.polyfit(cent_c[dyn], zero_phase_lowpass(ay_rot, dt, 10.0)[dyn], 1)[0], 0.80, 1.15))
                ay_scale = (1.0 / max(slope_y, 0.80)) * float(imu_cfg.get("scale_ay", 1.0))
                ay_cal = ay_rot * ay_scale
                ay_off = float(np.mean(ay_cal[dyn]) - np.mean(cent_ay[dyn]))
                ay_cal = np.where(vx_arr > 0.35, ay_cal - ay_off, ay_cal)
            else:
                ay_cal = ay_rot * float(imu_cfg.get("scale_ay", 1.0))
            imu_mode = f"yaw({yaw_deg:+.2f}deg)+adv(-{adv_samples*10}ms)+lever({x_lever:.2f}m)"
        else:
            ax_cal, ay_cal = ax_0, ay_0
    else:
        ax_cal = np.zeros_like(t_grid)
        ay_cal = np.zeros_like(t_grid)

    df["ax_raw_ms2"] = ax_cal
    df["ay_raw_ms2"] = ay_cal
    df["ax_ms2"] = zero_phase_lowpass(ax_cal, dt, cutoff_hz, butter_order)
    df["ay_ms2"] = zero_phase_lowpass(ay_cal, dt, cutoff_hz, butter_order)

    # 7. recorte de tiempos muertos y clasificación pista vs paddock_slow
    active = df["vx_ms"].to_numpy() >= min_active_vx
    df["is_active"] = active
    vx_p95 = float(np.percentile(df["vx_ms"].to_numpy(), 95))
    ay_p95 = float(np.percentile(np.abs(df["ay_ms2"].to_numpy()), 95))
    is_track = (
        vx_p95 >= float(filt_cfg.get("paddock_vx_p95_thresh_ms", 8.5))
        and ay_p95 >= float(filt_cfg.get("paddock_ay_p95_thresh_ms2", 4.5))
    )

    meta = {
        "log_type": "track" if is_track else "paddock_slow",
        "vx_source": vx_src,
        "vx_p95_ms": vx_p95,
        "ay_p95_ms2": ay_p95,
        "emi_spikes_filtered": emi_count,
        "steer_zero_deg": steer_zero_deg,
        "imu_mode": imu_mode,
        "active_samples": int(np.sum(active)),
        "total_samples": len(df),
    }
    return df, meta
