#!/usr/bin/env python3
"""Suite 09: Comparativa directa entre modelos (ej. sim_TeR26 vs sim_TeR27)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from .style import C_TER26, C_TER27, save_fig


def generate_cross_vehicle(out_dir: Path, all_runs: Dict[str, Dict[str, Any]]) -> None:
    """Genera comparativas de envolvente g-g, ganancia de guiñada y prestaciones entre modelos."""
    if len(all_runs) < 2:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    vids = sorted(all_runs.keys())
    colors = [C_TER26, C_TER27, "#059669", "#7C3AED"]

    # 1. comparativa de envolvente g-g y distribución de acel lateral
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, vid in enumerate(vids):
        data = all_runs[vid]
        lbl = data["payload"].get("legend_label", f"sim_{vid}")
        c = colors[i % len(colors)]
        runs = data["runs"]
        track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
        df_s = track_runs[0]["sim_open"]
        m_v = df_s["vx_sim"].to_numpy() > 3.0
        axes[0].scatter(df_s["ay_sim"].to_numpy()[m_v] / 9.80665, df_s["ax_sim"].to_numpy()[m_v] / 9.80665, s=5, alpha=0.35, color=c, label=lbl)
        axes[1].hist(np.abs(df_s["ay_sim"].to_numpy()[m_v]) / 9.80665, bins=30, alpha=0.45, color=c, density=True, label=lbl)

    axes[0].set_xlabel("ay (g)")
    axes[0].set_ylabel("ax (g)")
    axes[0].set_title("Envolvente g-g Simulada")
    axes[0].legend()
    axes[1].set_xlabel("|ay| (g)")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title("Distribución de Carga Lateral |ay|")
    axes[1].legend()
    fig.suptitle("Comparativa Multi-Vehículo: Envolvente Dinámica", fontweight="bold")
    save_fig(fig, out_dir / "cross_vehicle_gg_envelope.png")

    # 2. comparativa de correlación (r horizonte 10s y lazo abierto) y kus
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sigs = [("vx_ms", "vx"), ("ax_ms2", "ax"), ("ay_ms2", "ay"), ("wz_rads", "wz")]
    x = np.arange(len(sigs))
    w = 0.35
    for i, vid in enumerate(vids[:2]):
        p = all_runs[vid]["payload"]
        lbl = p.get("legend_label", f"sim_{vid}")
        c = colors[i % len(colors)]
        r_h = [p.get("pooled_horizon", {}).get(k, {}).get("r", 0.0) for k, _ in sigs]
        r_o = [p.get("pooled_open", {}).get(k, {}).get("r", 0.0) for k, _ in sigs]
        off = (i - 0.5) * w
        axes[0].bar(x + off, r_h, width=w, color=c, label=f"{lbl} (Horiz. 10s)")
        axes[1].bar(x + off, r_o, width=w, color=c, label=f"{lbl} (Lazo Abierto)")

    for ax, title in [(axes[0], "Pearson r — Horizonte 10 s"), (axes[1], "Pearson r — Lazo Abierto")]:
        ax.axhline(0.90, color="#059669", ls="--", lw=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([lb for _, lb in sigs])
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Comparativa Multi-Vehículo: Precisión de Correlación", fontweight="bold")
    save_fig(fig, out_dir / "cross_vehicle_metrics.png")
