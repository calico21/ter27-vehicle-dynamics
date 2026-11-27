#!/usr/bin/env python3
"""Suite 01: Dinámica global (series temporales, resumen r/R²/RMSE y evolución baseline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .style import C_ACCENT, C_HORIZON, C_MUTED, C_OPEN, C_REAL, save_fig


def generate(out_dir: Path, runs: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_lbl = payload.get("legend_label", "sim_TeR")
    disp = payload.get("display_name", "TeR")

    # 1. series temporales completas x tanda (máx 60s pa claridad visual en tandas largas)
    for r in runs:
        df_r = r["real"]
        df_h = r["sim_horizon"]
        df_o = r["sim_open"]
        t = df_r["t_s"].to_numpy()
        # elegir ventana de mayor actividad dinámica (hasta 45 s)
        if len(t) > 4500:
            vx = df_r["vx_ms"].to_numpy()
            roll_e = np.convolve(vx, np.ones(4500) / 4500.0, mode="valid")
            i0 = int(np.argmax(roll_e))
            sl = slice(i0, i0 + 4500)
        else:
            sl = slice(0, len(t))

        fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
        specs = [
            ("vx_ms", "vx_sim", "Velocidad vx (m/s)"),
            ("ax_ms2", "ax_sim", "Acel. long. ax (m/s²)"),
            ("ay_ms2", "ay_sim", "Acel. lat. ay (m/s²)"),
            ("wz_rads", "wz_sim", "Guiñada wz (rad/s)"),
            ("wheel_rpm_rl", "wheel_rpm_rl_sim", "RPM rueda RL"),
        ]
        for ax, (cr, cs, ylabel) in zip(axes, specs):
            ax.plot(t[sl], df_r[cr].to_numpy()[sl], color=C_REAL, lw=1.3, label="Real CAN", alpha=0.9)
            ax.plot(t[sl], df_h[cs].to_numpy()[sl], color=C_HORIZON, lw=1.1, label=f"{legend_lbl} (Horizonte 10s)", alpha=0.85)
            ax.plot(t[sl], df_o[cs].to_numpy()[sl], color=C_OPEN, lw=1.0, ls="--", label=f"{legend_lbl} (Lazo Abierto)", alpha=0.8)
            ax.set_ylabel(ylabel)
        axes[0].legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("Tiempo (s)")
        fig.suptitle(f"{disp} — Series Temporales: {r['name']}", fontweight="bold")
        save_fig(fig, out_dir / f"timeseries_{r['name']}.png")

    # 2. resumen global de r, r² y rmse x tanda y agrupado
    track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
    names = [r["name"][:18] for r in track_runs] + ["AGRUPADO"]
    x = np.arange(len(names))
    sigs = [("vx_ms", "vx"), ("ax_ms2", "ax"), ("ay_ms2", "ay"), ("wz_rads", "wz")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    w = 0.19
    for idx, (sk, slab) in enumerate(sigs):
        r_vals = [r["metrics_horizon"].get(sk, {}).get("r", 0.0) for r in track_runs] + [
            payload.get("pooled_horizon", {}).get(sk, {}).get("r", 0.0)
        ]
        r2_vals = [max(-0.5, r["metrics_horizon"].get(sk, {}).get("r2", 0.0)) for r in track_runs] + [
            max(-0.5, payload.get("pooled_horizon", {}).get(sk, {}).get("r2", 0.0))
        ]
        rmse_vals = [r["metrics_horizon"].get(sk, {}).get("rmse", 0.0) for r in track_runs] + [
            payload.get("pooled_horizon", {}).get(sk, {}).get("rmse", 0.0)
        ]
        off = (idx - 1.5) * w
        axes[0].bar(x + off, r_vals, width=w, label=slab)
        axes[1].bar(x + off, r2_vals, width=w, label=slab)
        axes[2].bar(x + off, rmse_vals, width=w, label=slab)

    axes[0].axhline(0.90, color=C_ACCENT, ls="--", lw=1.0, label="Objetivo r=0.90")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Correlación Pearson r (Horizonte 10s)")
    axes[1].set_ylim(-0.2, 1.05)
    axes[1].set_title("Coeficiente R² (Horizonte 10s)")
    axes[2].set_title("RMSE por señal")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.legend(fontsize=7)
    fig.suptitle(f"{disp} — Resumen Global de Correlación ({legend_lbl})", fontweight="bold")
    save_fig(fig, out_dir / "global_metrics_summary.png")

    # 3. comparativa baseline sin calibrar vs calibrado actual (si existe baseline_reference)
    base_ref = payload.get("baseline_reference")
    if isinstance(base_ref, dict) and base_ref:
        pooled_h = payload.get("pooled_horizon", {})
        labels = ["vx", "ax", "ay", "wz"]
        keys = ["vx_ms", "ax_ms2", "ay_ms2", "wz_rads"]
        r_before = [float(base_ref.get(f"r_{lb}", 0.0)) for lb in labels]
        r_after = [float(pooled_h.get(k, {}).get("r", 0.0)) for k in keys]
        rmse_before = [float(base_ref.get(f"rmse_{lb}", 1.0)) for lb in labels]
        rmse_after = [float(pooled_h.get(k, {}).get("rmse", 0.0)) for k in keys]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        xb = np.arange(len(labels))
        axes[0].bar(xb - 0.18, r_before, width=0.35, color=C_MUTED, label="Baseline sin calibrar")
        axes[0].bar(xb + 0.18, r_after, width=0.35, color=C_HORIZON, label=f"{legend_lbl} calibrado")
        axes[0].axhline(0.90, color=C_ACCENT, ls="--", lw=1.0)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Evolución Pearson r")
        axes[0].set_xticks(xb)
        axes[0].set_xticklabels(labels)
        axes[0].legend()

        axes[1].bar(xb - 0.18, rmse_before, width=0.35, color=C_MUTED, label="Baseline sin calibrar")
        axes[1].bar(xb + 0.18, rmse_after, width=0.35, color=C_OPEN, label=f"{legend_lbl} calibrado")
        axes[1].set_title("Reducción de RMSE")
        axes[1].set_xticks(xb)
        axes[1].set_xticklabels(labels)
        axes[1].legend()
        fig.suptitle(f"{disp} — Evolución Baseline vs Calibrado", fontweight="bold")
        save_fig(fig, out_dir / "baseline_vs_calibrated.png")
