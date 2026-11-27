#!/usr/bin/env python3
"""Suite 05: Neumáticos y envolvente (ángulos de deriva, Fy(alpha) por eje y utilización combinada)."""

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
    m_v = df_s["vx_sim"].to_numpy() > 3.0

    # 1. ángulos de deriva x eje (alpha_f, alpha_r) y curvas de eje fy(alpha)
    alpha_f = 0.5 * (df_s["alpha_fl_deg"].to_numpy() + df_s["alpha_fr_deg"].to_numpy())
    alpha_r = 0.5 * (df_s["alpha_rl_deg"].to_numpy() + df_s["alpha_rr_deg"].to_numpy())
    fy_f = df_s["fy_fl"].to_numpy() + df_s["fy_fr"].to_numpy()
    fy_r = df_s["fy_rl"].to_numpy() + df_s["fy_rr"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sl = slice(0, min(len(t), 3500))
    axes[0].plot(t[sl], alpha_f[sl], color=C_HORIZON, lw=1.0, label="α delantero (°)")
    axes[0].plot(t[sl], alpha_r[sl], color=C_OPEN, lw=1.0, label="α trasero (°)")
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("Ángulo de deriva α (°)")
    axes[0].set_title("Ángulos de Deriva por Eje")
    axes[0].legend()

    if np.any(m_v):
        axes[1].scatter(alpha_f[m_v], fy_f[m_v], s=4, alpha=0.28, color=C_HORIZON, label="Eje delantero Fy_f")
        axes[1].scatter(alpha_r[m_v], fy_r[m_v], s=4, alpha=0.28, color=C_OPEN, label="Eje trasero Fy_r")
    axes[1].set_xlabel("Ángulo de deriva α (°)")
    axes[1].set_ylabel("Fuerza lateral de eje Fy (N)")
    axes[1].set_title("Curvas de Fuerza Lateral por Eje")
    axes[1].legend()
    fig.suptitle(f"{disp} ({legend_lbl}) — Deriva y Rigidez de Paso por Curva", fontweight="bold")
    save_fig(fig, out_dir / "axle_slip_and_fy_curves.png")

    # 2. utilización combinada de adherencia x rueda a lo largo de la vuelta
    fig, ax = plt.subplots(figsize=(13, 4.8))
    mu_ref = 1.45
    for col, lbl, c in [("fl", "FL", C_FL), ("fr", "FR", C_FR), ("rl", "RL", C_RL), ("rr", "RR", C_RR)]:
        fx = df_s[f"fx_{col}"].to_numpy()[sl]
        fy = df_s[f"fy_{col}"].to_numpy()[sl]
        fz = np.maximum(df_s[f"fz_{col}"].to_numpy()[sl], 120.0)
        util = 100.0 * np.sqrt(fx ** 2 + fy ** 2) / (mu_ref * fz)
        ax.plot(t[sl], np.clip(util, 0.0, 130.0), lw=0.95, color=c, label=f"Uso {lbl} (%)")
    ax.axhline(100.0, color="k", ls="--", lw=0.9, label="Límite nominal 100%")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Utilización combinada (%)")
    ax.set_title(f"{disp} ({legend_lbl}) — Utilización Combinada de Adherencia por Rueda")
    ax.legend(ncol=5)
    save_fig(fig, out_dir / "combined_grip_utilization.png")
