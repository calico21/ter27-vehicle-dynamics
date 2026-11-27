#!/usr/bin/env python3
"""
Motor de métricas de correlación Sim-to-Real:
Pearson r, R², RMSE, MAE, NRMSE, sesgo, lag óptimo (-200..+200 ms),
coherencia espectral, gradiente de subviraje Kus y desglose por fase de curva.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal import coherence


def compute_signal_metrics(real: np.ndarray, sim: np.ndarray, dt: float = 0.01) -> Dict[str, float]:
    """Calcula batería completa de métricas entre señal real y simulada."""
    mask = np.isfinite(real) & np.isfinite(sim)
    if int(np.sum(mask)) < 20:
        return {
            "r": float("nan"), "r2": float("nan"), "rmse": float("nan"),
            "mae": float("nan"), "nrmse_pct": float("nan"), "bias": float("nan"),
            "optimal_lag_ms": 0.0, "n": int(np.sum(mask)),
        }
    r_arr = real[mask]
    s_arr = sim[mask]
    diff = s_arr - r_arr
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    span = max(float(np.percentile(r_arr, 98) - np.percentile(r_arr, 2)), 1e-6)
    nrmse_pct = 100.0 * rmse / span

    std_r = float(np.std(r_arr))
    std_s = float(np.std(s_arr))
    if std_r > 1e-9 and std_s > 1e-9:
        pearson_r = float(np.corrcoef(r_arr, s_arr)[0, 1])
    else:
        pearson_r = 0.0

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((r_arr - np.mean(r_arr)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

    # barrido de lag óptimo (-200 ms a +200 ms)
    max_lag = min(20, len(r_arr) // 5)
    best_lag = 0
    best_c = -2.0
    if max_lag >= 1 and std_r > 1e-9 and std_s > 1e-9:
        r_n = (r_arr - np.mean(r_arr)) / std_r
        s_n = (s_arr - np.mean(s_arr)) / std_s
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                c = float(np.mean(r_n * s_n))
            elif lag > 0:
                c = float(np.mean(r_n[lag:] * s_n[:-lag]))
            else:
                c = float(np.mean(r_n[:lag] * s_n[-lag:]))
            if c > best_c:
                best_c = c
                best_lag = lag

    return {
        "r": pearson_r,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "nrmse_pct": nrmse_pct,
        "bias": bias,
        "optimal_lag_ms": float(best_lag * dt * 1000.0),
        "n": int(len(r_arr)),
    }


def compute_kus(vx: np.ndarray, ay: np.ndarray, wz: np.ndarray, steer_deg: np.ndarray, wheelbase_m: float = 1.55) -> Tuple[float, bool]:
    """Estima el gradiente de subviraje Kus (deg/g) por regresión lineal en apoyo."""
    mask = np.isfinite(vx) & np.isfinite(ay) & np.isfinite(wz) & np.isfinite(steer_deg) & (vx > 4.0) & (np.abs(ay) > 1.5)
    if int(np.sum(mask)) < 30:
        return 0.0, False
    ay_g = ay[mask] / 9.80665
    ack_deg = np.degrees(wheelbase_m * wz[mask] / vx[mask])
    us_deg = steer_deg[mask] - ack_deg
    slope = float(np.polyfit(ay_g, us_deg, 1)[0])
    return slope, True


def compute_corner_phase_errors(
    vx: np.ndarray, ax: np.ndarray, ay: np.ndarray, wz: np.ndarray,
    ax_sim: np.ndarray, ay_sim: np.ndarray, wz_sim: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Desglosa RMSE de ay, ax, wz en Entrada (frenada+giro), Vértice (apoyo) y Salida (tracción)."""
    moving = vx > 3.0
    lat = np.abs(ay) > 2.5
    entry = moving & lat & (ax < -1.0)
    apex = moving & lat & (np.abs(ax) <= 1.0)
    exit_ph = moving & lat & (ax > 1.0)

    out: Dict[str, Dict[str, float]] = {}
    for name, m in [("entrada", entry), ("vertice", apex), ("salida", exit_ph)]:
        n = int(np.sum(m))
        if n < 10:
            out[name] = {"n": n, "rmse_ay": float("nan"), "rmse_ax": float("nan"), "rmse_wz": float("nan")}
        else:
            out[name] = {
                "n": n,
                "rmse_ay": float(np.sqrt(np.mean((ay_sim[m] - ay[m]) ** 2))),
                "rmse_ax": float(np.sqrt(np.mean((ax_sim[m] - ax[m]) ** 2))),
                "rmse_wz": float(np.sqrt(np.mean((wz_sim[m] - wz[m]) ** 2))),
            }
    return out


def compute_spectral_coherence(u: np.ndarray, y: np.ndarray, fs_hz: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula coherencia cuadrática gamma^2(f) entre entrada u y salida y."""
    mask = np.isfinite(u) & np.isfinite(y)
    if int(np.sum(mask)) < 256:
        return np.array([0.0]), np.array([0.0])
    f, cxy = coherence(u[mask], y[mask], fs=fs_hz, nperseg=min(512, int(np.sum(mask))))
    return f, cxy
