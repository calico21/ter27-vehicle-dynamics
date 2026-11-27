#!/usr/bin/env python3
"""Suite 08: Espectro y frecuencia (PSD/FFT 0-50 Hz, función de transferencia y coherencia γ²)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import coherence, csd, welch

from .style import C_HORIZON, C_OPEN, C_REAL, save_fig


def generate(out_dir: Path, runs: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_lbl = payload.get("legend_label", "sim_TeR")
    disp = payload.get("display_name", "TeR")

    track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
    r0 = track_runs[0]
    df_r, df_s = r0["real"], r0["sim_open"]
    fs = 100.0
    nperseg = min(512, len(df_r))

    # 1. psd / fft de ax, ay, wz y vel de rueda rl (0-50 hz) sobre señal cruda
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        (axes[0, 0], "ax_raw_ms2", "ax_sim", "PSD Acel. Longitudinal ax"),
        (axes[0, 1], "ay_raw_ms2", "ay_sim", "PSD Acel. Lateral ay"),
        (axes[1, 0], "wz_raw_rads", "wz_sim", "PSD Velocidad de Guiñada wz"),
        (axes[1, 1], "wheel_rpm_rl_raw", "wheel_rpm_rl_sim", "PSD Rueda Trasera RL"),
    ]
    for ax, cr, cs, title in specs:
        ur = df_r[cr].to_numpy() if cr in df_r.columns else df_r[cr.replace("_raw", "")].to_numpy()
        us = df_s[cs].to_numpy()
        if len(ur) >= 64:
            fr, pr = welch(ur, fs=fs, nperseg=nperseg)
            fs_arr, ps = welch(us, fs=fs, nperseg=nperseg)
            ax.semilogy(fr, np.maximum(pr, 1e-10), color=C_REAL, lw=1.2, label="Real CAN (cruda)")
            ax.semilogy(fs_arr, np.maximum(ps, 1e-10), color=C_HORIZON, lw=1.1, label=legend_lbl)
        ax.set_xlim(0, 50)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(title)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"{disp} ({legend_lbl}) — Densidad Espectral de Potencia (0-50 Hz)", fontweight="bold")
    save_fig(fig, out_dir / "psd_signals_0_50hz.png")

    # 2. función de transferencia (ganancia/fase) y coherencia γ²(f) entre volante δ y wz / ay
    u = np.radians(df_r["steer_deg"].to_numpy())
    y_wz_r = df_r["wz_rads"].to_numpy()
    y_wz_s = df_s["wz_sim"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if len(u) >= 256:
        f, p_uu = welch(u, fs=fs, nperseg=nperseg)
        _, p_uy_r = csd(u, y_wz_r, fs=fs, nperseg=nperseg)
        _, p_uy_s = csd(u, y_wz_s, fs=fs, nperseg=nperseg)
        h_r = p_uy_r / np.maximum(p_uu, 1e-12)
        h_s = p_uy_s / np.maximum(p_uu, 1e-12)
        _, coh_r = coherence(u, y_wz_r, fs=fs, nperseg=nperseg)
        _, coh_s = coherence(u, y_wz_s, fs=fs, nperseg=nperseg)

        axes[0].plot(f, np.abs(h_r), color=C_REAL, lw=1.3, label="Real |wz/δ|")
        axes[0].plot(f, np.abs(h_s), color=C_HORIZON, lw=1.2, label=f"{legend_lbl} |wz/δ|")
        axes[0].set_xlim(0, 12)
        axes[0].set_title("Ganancia Función de Transferencia")
        axes[0].set_ylabel("|H(f)| (rad/s / rad)")
        axes[0].legend()

        axes[1].plot(f, np.degrees(np.angle(h_r)), color=C_REAL, lw=1.3, label="Real fase")
        axes[1].plot(f, np.degrees(np.angle(h_s)), color=C_HORIZON, lw=1.2, label=f"{legend_lbl} fase")
        axes[1].set_xlim(0, 12)
        axes[1].set_title("Fase de Respuesta en Guiñada")
        axes[1].set_ylabel("Fase (°)")
        axes[1].legend()

        axes[2].plot(f, coh_r, color=C_REAL, lw=1.3, label="Coherencia Real (δ → wz)")
        axes[2].plot(f, coh_s, color=C_OPEN, lw=1.2, label=f"Coherencia {legend_lbl} (δ → wz)")
        axes[2].set_xlim(0, 12)
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title("Coherencia Espectral γ²(f)")
        axes[2].set_ylabel("γ²(f)")
        axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Frecuencia (Hz)")
    fig.suptitle(f"{disp} ({legend_lbl}) — Respuesta en Frecuencia Volante → Guiñada", fontweight="bold")
    save_fig(fig, out_dir / "transfer_function_and_coherence.png")
