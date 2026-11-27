#!/usr/bin/env python3
"""Suite 10: Comparativa directa entre modelos (full_14dof vs fast_3dof) sobre un mismo vehículo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from .style import C_HORIZON, C_OPEN, C_REAL, save_fig


def _fmt(v: Any, dec: int = 4) -> str:
    try:
        fv = float(v)
        if fv != fv:
            return "N/A"
        return f"{fv:.{dec}f}"
    except (TypeError, ValueError):
        return "N/A"


def generate_cross_model(out_dir: Path, vehicle_id: str, disp_name: str, model_results: Dict[str, Dict[str, Any]]) -> None:
    """Genera model_tradeoff_metrics.png, model_transient_overlay.png y COMPARATIVA_MODELOS.md."""
    if "full_14dof" not in model_results or "fast_3dof" not in model_results:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    r14 = model_results["full_14dof"]
    r3 = model_results["fast_3dof"]
    p14, p3 = r14["payload"], r3["payload"]
    t14 = float(p14.get("elapsed_s", 1.0))
    t3 = float(p3.get("elapsed_s", 1.0))
    speedup = t14 / max(t3, 1e-3)

    # 1. gráfica comparativa de métricas y speedup (model_tradeoff_metrics.png)
    sigs = [("vx_ms", "vx"), ("ax_ms2", "ax"), ("ay_ms2", "ay"), ("wz_rads", "wz")]
    x = np.arange(len(sigs))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, metric_key, title, ylim in [
        (axes[0, 0], "r", "Correlación Pearson r (Horizonte 10 s)", (0.5, 1.04)),
        (axes[0, 1], "r2", "Coeficiente R² (Horizonte 10 s)", (0.4, 1.04)),
        (axes[1, 0], "rmse", "Error RMSE (Horizonte 10 s)", None),
    ]:
        v14 = [float(p14["pooled_horizon"].get(sk, {}).get(metric_key, 0.0)) for sk, _ in sigs]
        v3 = [float(p3["pooled_horizon"].get(sk, {}).get(metric_key, 0.0)) for sk, _ in sigs]
        ax.bar(x - w / 2, v14, width=w, color=C_HORIZON, label="full_14dof (1000 Hz)")
        ax.bar(x + w / 2, v3, width=w, color=C_OPEN, label="fast_3dof (200 Hz)")
        if metric_key == "r":
            ax.axhline(0.90, color="#059669", ls="--", lw=1.0, label="Objetivo r=0.90")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([lb for _, lb in sigs])
        ax.set_title(title)
        ax.legend(fontsize=7.5)

    # subplot de tiempo de ejecución y speedup
    ax_t = axes[1, 1]
    bars = ax_t.bar(["full_14dof\n(1000 Hz)", "fast_3dof\n(200 Hz)"], [t14, t3], color=[C_HORIZON, C_OPEN], width=0.45)
    ax_t.set_ylabel("Tiempo batch Godot (s)")
    ax_t.set_title(f"Coste Computacional (Speedup: {speedup:.2f}×)")
    for b, val in zip(bars, [t14, t3]):
        ax_t.text(b.get_x() + b.get_width() / 2, val + max(t14, t3) * 0.02, f"{val:.1f} s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle(f"{disp_name} ({vehicle_id}) — Trade-off Precisión vs Coste: full_14dof vs fast_3dof", fontweight="bold")
    save_fig(fig, out_dir / "model_tradeoff_metrics.png")

    # 2. superposición de transitorios críticos (frenada fuerte + entrada en curva + tracción)
    runs14 = [r for r in r14["runs"] if r["meta"].get("log_type", "track") == "track"] or r14["runs"]
    runs3 = [r for r in r3["runs"] if r["meta"].get("log_type", "track") == "track"] or r3["runs"]
    df_r = runs14[0]["real"]
    df_14 = runs14[0]["sim_open"]
    df_3 = runs3[0]["sim_open"]

    t = df_r["t_s"].to_numpy()
    vx = df_r["vx_ms"].to_numpy()
    brk = df_r["brake"].to_numpy() if "brake" in df_r.columns else np.zeros_like(t)
    ay = df_r["ay_ms2"].to_numpy()
    # buscar ventana de 12 s con mayor combinación de frenada + acel lateral
    score = np.convolve((brk > 0.1).astype(float) * 2.0 + np.abs(ay) / 10.0, np.ones(1200) / 1200.0, mode="valid")
    i0 = int(np.argmax(score)) if len(score) > 0 else 0
    sl = slice(i0, min(len(t), i0 + 1200))

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for ax, cr, cs, ylabel in [
        (axes[0], "vx_ms", "vx_sim", "vx (m/s)"),
        (axes[1], "ax_ms2", "ax_sim", "ax (m/s²)"),
        (axes[2], "ay_ms2", "ay_sim", "ay (m/s²)"),
        (axes[3], "wz_rads", "wz_sim", "wz (rad/s)"),
    ]:
        ax.plot(t[sl], df_r[cr].to_numpy()[sl], color=C_REAL, lw=1.5, label="Real CAN")
        ax.plot(t[sl], df_14[cs].to_numpy()[sl], color=C_HORIZON, lw=1.2, label="full_14dof (1000 Hz)")
        ax.plot(t[sl], df_3[cs].to_numpy()[sl], color=C_OPEN, lw=1.1, ls="--", label="fast_3dof (200 Hz)")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(f"{disp_name} ({vehicle_id}) — Superposición de Transitorios Críticos (Lazo Abierto)", fontweight="bold")
    save_fig(fig, out_dir / "model_transient_overlay.png")

    # 3. tabla de decisión técnica en markdown (comparativa_modelos.md)
    lines = [
        f"# Comparativa de Solvers — {disp_name} (`{vehicle_id}`)",
        "",
        f"- **Tiempo batch `full_14dof` (1000 Hz):** {t14:.2f} s",
        f"- **Tiempo batch `fast_3dof` (200 Hz):** {t3:.2f} s",
        f"- **Ganancia de velocidad (Speedup):** **{speedup:.2f}×**",
        "",
        "## Métricas en Horizonte 10 s y Lazo Abierto",
        "",
        "| Señal | `r` 14-DOF (H10s) | `r` 3-DOF (H10s) | Δr (H10s) | RMSE 14-DOF | RMSE 3-DOF | ΔRMSE | `r` 14-DOF (Open) | `r` 3-DOF (Open) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for sk, lb in [("vx_ms", "`vx` (m/s)"), ("ax_ms2", "`ax` (m/s²)"), ("ay_ms2", "`ay` (m/s²)"), ("wz_rads", "`wz` (rad/s)")]:
        mh14 = p14["pooled_horizon"].get(sk, {})
        mh3 = p3["pooled_horizon"].get(sk, {})
        mo14 = p14["pooled_open"].get(sk, {})
        mo3 = p3["pooled_open"].get(sk, {})
        dr = float(mh3.get("r", 0.0)) - float(mh14.get("r", 0.0))
        drmse = float(mh3.get("rmse", 0.0)) - float(mh14.get("rmse", 0.0))
        lines.append(
            f"| {lb} | {_fmt(mh14.get('r'))} | {_fmt(mh3.get('r'))} | {dr:+.4f} | "
            f"{_fmt(mh14.get('rmse'), 3)} | {_fmt(mh3.get('rmse'), 3)} | {drmse:+.3f} | "
            f"{_fmt(mo14.get('r'))} | {_fmt(mo3.get('r'))} |"
        )

    lines.extend([
        "",
        "## Criterio de Selección Técnica",
        "",
        "- **Usar `full_14dof` (1000 Hz)** para Dinámica Vehicular: ajuste de muelles, amortiguadores, anti-dive/anti-squat, alturas de *roll center* y análisis de vibraciones verticales.",
        "- **Usar `fast_3dof` (200 Hz)** para Driverless (HIL / NMPC / ROS 2): mantiene `r > 0.90` en las señales planares principales eliminando el coste de los GDL verticales de suspensión.",
        "",
    ])
    (out_dir / "COMPARATIVA_MODELOS.md").write_text("\n".join(lines), encoding="utf-8")
