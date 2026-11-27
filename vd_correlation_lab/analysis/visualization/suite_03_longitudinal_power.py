#!/usr/bin/env python3
"""Suite 03: Dinámica longitudinal, velocidades de rueda, slip ratios, pedales y lift-off."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .style import C_FL, C_FR, C_HORIZON, C_OPEN, C_REAL, C_RL, C_RR, save_fig


def generate(out_dir: Path, runs: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_lbl = payload.get("legend_label", "sim_TeR")
    disp = payload.get("display_name", "TeR")
    wheel_r = float(payload.get("wheel_radius_m", 0.2032))

    track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
    r0 = track_runs[0]
    df_r, df_s = r0["real"], r0["sim_open"]
    t = df_r["t_s"].to_numpy()
    sl = slice(0, min(len(t), 4000))

    # 1. vels lineales de las 4 ruedas vs vx chasis
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    rpm_to_ms = (2.0 * np.pi / 60.0) * wheel_r
    for col, lbl, c in [("fl", "FL", C_FL), ("fr", "FR", C_FR), ("rl", "RL", C_RL), ("rr", "RR", C_RR)]:
        if f"wheel_rpm_{col}" in df_r.columns:
            axes[0].plot(t[sl], df_r[f"wheel_rpm_{col}"].to_numpy()[sl] * rpm_to_ms, lw=0.9, color=c, label=f"Real {lbl}")
        axes[1].plot(t[sl], df_s[f"wheel_rpm_{col}_sim"].to_numpy()[sl] * rpm_to_ms, lw=0.9, color=c, label=f"{legend_lbl} {lbl}")
    axes[0].plot(t[sl], df_r["vx_ms"].to_numpy()[sl], color=C_REAL, lw=1.4, ls="--", label="vx Chasis Real")
    axes[1].plot(t[sl], df_s["vx_sim"].to_numpy()[sl], color=C_REAL, lw=1.4, ls="--", label=f"vx {legend_lbl}")
    axes[0].set_ylabel("Velocidad (m/s)")
    axes[0].set_title("Telemetría Real CAN — Ruedas vs vx")
    axes[0].legend(ncol=5, fontsize=7.5)
    axes[1].set_ylabel("Velocidad (m/s)")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_title(f"{legend_lbl} — Ruedas vs vx")
    axes[1].legend(ncol=5, fontsize=7.5)
    fig.suptitle(f"{disp} — Comparativa de Velocidades de Rueda ({r0['name']})", fontweight="bold")
    save_fig(fig, out_dir / "wheel_speeds_vs_vx.png")

    # 2. evolución de deslizamiento longitudinal kappa x rueda
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for col, lbl, c in [("fl", "FL", C_FL), ("fr", "FR", C_FR), ("rl", "RL", C_RL), ("rr", "RR", C_RR)]:
        ax.plot(t[sl], np.clip(df_s[f"kappa_{col}"].to_numpy()[sl], -0.35, 0.35), lw=1.0, color=c, label=f"κ {lbl}")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Slip ratio κ (-)")
    ax.set_title(f"{disp} ({legend_lbl}) — Evolución de Deslizamiento Longitudinal κ")
    ax.legend(ncol=4)
    save_fig(fig, out_dir / "longitudinal_slip_kappa.png")

    # 3. curvas de respuesta a pedales: ax vs apps (x trq_limit) y ax vs bpps
    thr_all = np.concatenate([r["real"]["throttle"].to_numpy() for r in track_runs])
    brk_all = np.concatenate([r["real"]["brake"].to_numpy() for r in track_runs])
    trq_all = np.concatenate([r["real"]["trq_limit_nm"].to_numpy() for r in track_runs])
    vx_all = np.concatenate([r["real"]["vx_ms"].to_numpy() for r in track_runs])
    ax_r_all = np.concatenate([r["real"]["ax_ms2"].to_numpy() for r in track_runs])
    ax_s_all = np.concatenate([r["sim_open"]["ax_sim"].to_numpy() for r in track_runs])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    m_drv = (vx_all > 3.0) & (thr_all > 0.05) & (brk_all < 0.05)
    if np.sum(m_drv) > 30:
        hi_trq = trq_all[m_drv] >= 100.0
        lo_trq = ~hi_trq
        if np.any(lo_trq):
            axes[0].scatter(thr_all[m_drv][lo_trq] * 100.0, ax_r_all[m_drv][lo_trq], s=5, alpha=0.25, color=C_RL, label="Real (<100 Nm)")
        if np.any(hi_trq):
            axes[0].scatter(thr_all[m_drv][hi_trq] * 100.0, ax_r_all[m_drv][hi_trq], s=5, alpha=0.25, color=C_REAL, label="Real (≥100 Nm)")
        axes[0].scatter(thr_all[m_drv] * 100.0, ax_s_all[m_drv], s=5, alpha=0.2, color=C_HORIZON, label=legend_lbl)
    axes[0].set_xlabel("APPS (%)")
    axes[0].set_ylabel("ax (m/s²)")
    axes[0].set_title("Respuesta en Tracción (ax vs APPS)")
    axes[0].legend(fontsize=7.5)

    m_brk = (vx_all > 3.0) & (brk_all > 0.05)
    if np.sum(m_brk) > 20:
        axes[1].scatter(brk_all[m_brk] * 100.0, ax_r_all[m_brk], s=6, alpha=0.35, color=C_REAL, label="Real CAN")
        axes[1].scatter(brk_all[m_brk] * 100.0, ax_s_all[m_brk], s=6, alpha=0.35, color=C_OPEN, label=legend_lbl)
        axes[1].legend(fontsize=7.5)
    axes[1].set_xlabel("BPPS normalizado (%)")
    axes[1].set_ylabel("ax (m/s²)")
    axes[1].set_title("Eficiencia de Frenada (ax vs BPPS)")
    fig.suptitle(f"{disp} — Mapas de Respuesta a Pedales", fontweight="bold")
    save_fig(fig, out_dir / "pedal_response_maps.png")

    # 4. ventana de suelta de gas (lift-off regen) y trail braking
    thr0 = df_r["throttle"].to_numpy()
    brk0 = df_r["brake"].to_numpy()
    vx0 = df_r["vx_ms"].to_numpy()
    trans = (vx0 > 6.0) & (np.gradient(thr0) < -0.05)
    idx_cand = np.where(trans)[0]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if len(idx_cand) > 0:
        c0 = int(idx_cand[len(idx_cand) // 2])
        w0, w1 = max(0, c0 - 150), min(len(t), c0 + 250)
        ax.plot(t[w0:w1], df_r["ax_ms2"].to_numpy()[w0:w1], color=C_REAL, lw=1.4, label="Real ax (m/s²)")
        ax.plot(t[w0:w1], df_s["ax_sim"].to_numpy()[w0:w1], color=C_HORIZON, lw=1.3, label=f"{legend_lbl} ax (m/s²)")
        ax2 = ax.twinx()
        ax2.plot(t[w0:w1], thr0[w0:w1] * 100.0, color=C_FR, lw=0.9, ls=":", label="APPS (%)")
        ax2.plot(t[w0:w1], brk0[w0:w1] * 100.0, color=C_OPEN, lw=0.9, ls=":", label="BPPS (%)")
        ax2.set_ylabel("Pedal (%)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("ax (m/s²)")
    ax.set_title(f"{disp} — Transición Lift-Off Regen y Trail Braking")
    ax.legend(loc="lower left")
    save_fig(fig, out_dir / "liftoff_and_trail_braking.png")
