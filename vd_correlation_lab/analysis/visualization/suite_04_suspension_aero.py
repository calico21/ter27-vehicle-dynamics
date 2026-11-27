#!/usr/bin/env python3
"""Suite 04: Suspensión y aerodinámica (balanceo, cabeceo, Fz, LLTD, amortiguadores y downforce)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .style import C_FL, C_FR, C_HORIZON, C_OPEN, C_RL, C_RR, save_fig


def generate(out_dir: Path, runs: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_lbl = payload.get("legend_label", "sim_TeR")
    disp = payload.get("display_name", "TeR")

    track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
    df_s = track_runs[0]["sim_open"]
    t = df_s["t_s"].to_numpy()
    vx = df_s["vx_sim"].to_numpy()
    ax_s = df_s["ax_sim"].to_numpy()
    ay_s = df_s["ay_sim"].to_numpy()
    roll = df_s["roll_deg_sim"].to_numpy()
    pitch = df_s["pitch_deg_sim"].to_numpy()

    # 1. gradientes de balanceo (deg/g vs ay) y cabeceo (deg/g vs ax)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    m_v = vx > 3.0
    if np.sum(m_v) > 30:
        ay_g = ay_s[m_v] / 9.80665
        ax_g = ax_s[m_v] / 9.80665
        axes[0].scatter(ay_g, roll[m_v], s=5, alpha=0.3, color=C_HORIZON)
        if np.std(ay_g) > 0.05:
            kr = np.polyfit(ay_g, roll[m_v], 1)[0]
            xx = np.linspace(np.min(ay_g), np.max(ay_g), 50)
            axes[0].plot(xx, np.polyval(np.polyfit(ay_g, roll[m_v], 1), xx), color=C_OPEN, lw=1.4, label=f"Gradiente: {kr:.2f} °/g")
            axes[0].legend()
        axes[1].scatter(ax_g, pitch[m_v], s=5, alpha=0.3, color=C_HORIZON)
        if np.std(ax_g) > 0.05:
            kp = np.polyfit(ax_g, pitch[m_v], 1)[0]
            xx = np.linspace(np.min(ax_g), np.max(ax_g), 50)
            axes[1].plot(xx, np.polyval(np.polyfit(ax_g, pitch[m_v], 1), xx), color=C_OPEN, lw=1.4, label=f"Gradiente: {kp:.2f} °/g")
            axes[1].legend()
    axes[0].set_xlabel("ay (g)")
    axes[0].set_ylabel("Balanceo (°)")
    axes[0].set_title("Gradiente de Balanceo")
    axes[1].set_xlabel("ax (g)")
    axes[1].set_ylabel("Cabeceo (°)")
    axes[1].set_title("Gradiente de Cabeceo")
    fig.suptitle(f"{disp} ({legend_lbl}) — Cinemática de Chasis", fontweight="bold")
    save_fig(fig, out_dir / "roll_and_pitch_gradients.png")

    # 2. cargas verticales dinámicas fz y reparto lateral lltd del vs tras
    sl = slice(0, min(len(t), 3500))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for col, lbl, c in [("fl", "FL", C_FL), ("fr", "FR", C_FR), ("rl", "RL", C_RL), ("rr", "RR", C_RR)]:
        axes[0].plot(t[sl], df_s[f"fz_{col}"].to_numpy()[sl], lw=1.0, color=c, label=f"Fz {lbl}")
    axes[0].set_ylabel("Carga vertical Fz (N)")
    axes[0].set_title("Cargas Verticales Dinámicas en las 4 Ruedas")
    axes[0].legend(ncol=4)

    dfz_f = np.abs(df_s["fz_fr"].to_numpy() - df_s["fz_fl"].to_numpy())
    dfz_r = np.abs(df_s["fz_rr"].to_numpy() - df_s["fz_rl"].to_numpy())
    tot_lat = dfz_f + dfz_r
    lltd_f = np.where(tot_lat > 80.0, 100.0 * dfz_f / np.maximum(tot_lat, 1.0), np.nan)
    axes[1].plot(t[sl], lltd_f[sl], color=C_HORIZON, lw=1.1, label="LLTD Delantero (%)")
    axes[1].axhline(50.0, color="k", lw=0.8, ls=":")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("LLTD Delantero (%)")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_title("Evolución del Reparto Lateral de Transferencia de Carga (LLTD)")
    axes[1].legend()
    fig.suptitle(f"{disp} ({legend_lbl}) — Cargas Verticales y LLTD", fontweight="bold")
    save_fig(fig, out_dir / "vertical_loads_and_lltd.png")

    # 3. histograma de vels de amortiguador y carga aerodinámica vs vx²
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    v_damper = np.concatenate([
        df_s[f"damper_vel_{c}"].to_numpy()[m_v] * 1000.0 for c in ("fl", "fr", "rl", "rr")
    ]) if np.any(m_v) else np.zeros(10)
    v_damper = v_damper[np.abs(v_damper) < 600.0]
    axes[0].hist(v_damper, bins=50, color=C_HORIZON, alpha=0.8, edgecolor="white")
    axes[0].axvline(-50, color=C_OPEN, ls="--", lw=0.9, label="Baja/Alta vel (±50 mm/s)")
    axes[0].axvline(50, color=C_OPEN, ls="--", lw=0.9)
    axes[0].set_xlabel("Velocidad de amortiguador (mm/s)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Histograma de Velocidades de Amortiguador")
    axes[0].legend()

    downf = df_s["f_downforce_n"].to_numpy()
    axes[1].scatter(vx[m_v] ** 2, downf[m_v], s=5, alpha=0.35, color=C_HORIZON, label=legend_lbl)
    axes[1].set_xlabel("vx² (m²/s²)")
    axes[1].set_ylabel("Carga aerodinámica (N)")
    axes[1].set_title("Downforce frente a vx²")
    axes[1].legend()
    fig.suptitle(f"{disp} ({legend_lbl}) — Amortiguadores y Aerodinámica", fontweight="bold")
    save_fig(fig, out_dir / "damper_hist_and_downforce.png")
