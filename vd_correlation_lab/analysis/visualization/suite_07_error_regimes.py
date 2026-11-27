#!/usr/bin/env python3
"""Suite 07: Análisis de error y regímenes (mapas de calor 2D, histogramas, cross-corr lag y fases de curva)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .style import C_HORIZON, C_OPEN, C_REAL, save_fig


def _binned_rmse(x: np.ndarray, y: np.ndarray, err: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    grid = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
    for iy in range(len(y_edges) - 1):
        my = (y >= y_edges[iy]) & (y < y_edges[iy + 1])
        for ix in range(len(x_edges) - 1):
            m = my & (x >= x_edges[ix]) & (x < x_edges[ix + 1])
            if np.sum(m) >= 8:
                grid[iy, ix] = float(np.sqrt(np.mean(err[m] ** 2)))
    return grid


def generate(out_dir: Path, runs: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_lbl = payload.get("legend_label", "sim_TeR")
    disp = payload.get("display_name", "TeR")

    track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
    vx = np.concatenate([r["real"]["vx_ms"].to_numpy() for r in track_runs])
    ax_r = np.concatenate([r["real"]["ax_ms2"].to_numpy() for r in track_runs])
    ay_r = np.concatenate([r["real"]["ay_ms2"].to_numpy() for r in track_runs])
    wz_r = np.concatenate([r["real"]["wz_rads"].to_numpy() for r in track_runs])

    vx_s = np.concatenate([r["sim_open"]["vx_sim"].to_numpy() for r in track_runs])
    ax_s = np.concatenate([r["sim_open"]["ax_sim"].to_numpy() for r in track_runs])
    ay_s = np.concatenate([r["sim_open"]["ay_sim"].to_numpy() for r in track_runs])
    wz_s = np.concatenate([r["sim_open"]["wz_sim"].to_numpy() for r in track_runs])

    m_v = vx > 3.0
    e_ay = ay_s - ay_r
    e_ax = ax_s - ax_r
    e_wz = wz_s - wz_r
    e_vx = vx_s - vx

    # 1. mapas de calor 2d del error en (vx, |ay|) y (ax, ay)
    vx_edges = np.linspace(3.0, max(16.0, float(np.percentile(vx[m_v], 98)) if np.any(m_v) else 20.0), 10)
    ay_edges = np.linspace(0.0, max(12.0, float(np.percentile(np.abs(ay_r[m_v]), 98)) if np.any(m_v) else 15.0), 9)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, err, title in [
        (axes[0, 0], e_ay, "RMSE ay (m/s²) en (vx, |ay|)"),
        (axes[0, 1], e_ax, "RMSE ax (m/s²) en (vx, |ay|)"),
        (axes[1, 0], e_wz, "RMSE wz (rad/s) en (vx, |ay|)"),
        (axes[1, 1], e_vx, "RMSE vx (m/s) en (vx, |ay|)"),
    ]:
        g = _binned_rmse(vx[m_v], np.abs(ay_r[m_v]), err[m_v], vx_edges, ay_edges)
        im = ax.imshow(
            g, origin="lower", aspect="auto", cmap="inferno",
            extent=[vx_edges[0], vx_edges[-1], ay_edges[0], ay_edges[-1]],
        )
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("vx (m/s)")
        ax.set_ylabel("|ay| (m/s²)")
        ax.set_title(title)
    fig.suptitle(f"{disp} ({legend_lbl}) — Mapas de Calor de Error en Plano Operacional", fontweight="bold")
    save_fig(fig, out_dir / "error_heatmaps_vx_ay.png")

    # mapa 2d en plano (ax, ay)
    ax_edges = np.linspace(-14.0, 12.0, 10)
    ay_s_edges = np.linspace(-15.0, 15.0, 10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, err, title in [
        (axes[0], e_ay, "RMSE ay (m/s²) en plano (ay, ax)"),
        (axes[1], e_wz, "RMSE wz (rad/s) en plano (ay, ax)"),
    ]:
        g = _binned_rmse(ay_r[m_v], ax_r[m_v], err[m_v], ay_s_edges, ax_edges)
        im = ax.imshow(
            g, origin="lower", aspect="auto", cmap="inferno",
            extent=[ay_s_edges[0], ay_s_edges[-1], ax_edges[0], ax_edges[-1]],
        )
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("ay (m/s²)")
        ax.set_ylabel("ax (m/s²)")
        ax.set_title(title)
    fig.suptitle(f"{disp} ({legend_lbl}) — Error en Plano g-g (ax, ay)", fontweight="bold")
    save_fig(fig, out_dir / "error_heatmaps_ax_ay.png")

    # 2. distribución de residuos + ajuste normal y barrido de correlación cruzada (-200..+200 ms)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, err, lbl in [(axes[0], e_ay[m_v], "Residuo ay (m/s²)"), (axes[1], e_wz[m_v], "Residuo wz (rad/s)")]:
        if len(err) > 20:
            mu, sd = float(np.mean(err)), max(float(np.std(err)), 1e-6)
            ax.hist(err, bins=45, density=True, color=C_HORIZON, alpha=0.7)
            xx = np.linspace(mu - 3.5 * sd, mu + 3.5 * sd, 120)
            ax.plot(xx, norm.pdf(xx, mu, sd), color=C_OPEN, lw=1.4, label=f"N({mu:.2f}, {sd:.2f}²)")
            ax.legend()
        ax.set_xlabel(lbl)
        ax.set_ylabel("Densidad")
        ax.set_title(f"Distribución: {lbl}")

    # cross-correlation lag sweep (-200 ms a +200 ms)
    lags = np.arange(-20, 21)
    for rv, sv, lbl, c in [(ay_r, ay_s, "ay", C_HORIZON), (wz_r, wz_s, "wz", C_OPEN), (ax_r, ax_s, "ax", C_REAL)]:
        rn = (rv - np.mean(rv)) / max(np.std(rv), 1e-6)
        sn = (sv - np.mean(sv)) / max(np.std(sv), 1e-6)
        corrs = []
        for lg in lags:
            if lg == 0:
                corrs.append(float(np.mean(rn * sn)))
            elif lg > 0:
                corrs.append(float(np.mean(rn[lg:] * sn[:-lg])))
            else:
                corrs.append(float(np.mean(rn[:lg] * sn[-lg:])))
        axes[2].plot(lags * 10.0, corrs, lw=1.3, color=c, label=lbl)
    axes[2].axvline(0.0, color="k", ls=":", lw=0.8)
    axes[2].set_xlabel("Desfase τ (ms)")
    axes[2].set_ylabel("Correlación cruzada")
    axes[2].set_title("Barrido Cross-Correlation (-200 a +200 ms)")
    axes[2].legend()
    fig.suptitle(f"{disp} ({legend_lbl}) — Residuos y Sincronización Temporal", fontweight="bold")
    save_fig(fig, out_dir / "residuals_and_lag_sweep.png")

    # 3. desglose x fase de curva (entrada / vértice / salida)
    phases = payload.get("corner_phases_open", {})
    ph_names = ["entrada", "vertice", "salida"]
    ph_labels = ["Entrada\n(Frenada+Giro)", "Vértice\n(Apoyo)", "Salida\n(Tracción)"]
    x = np.arange(len(ph_names))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    w = 0.25
    ax.bar(x - w, [phases.get(k, {}).get("rmse_ay", 0.0) for k in ph_names], width=w, color=C_HORIZON, label="RMSE ay (m/s²)")
    ax.bar(x, [phases.get(k, {}).get("rmse_ax", 0.0) for k in ph_names], width=w, color=C_OPEN, label="RMSE ax (m/s²)")
    ax.bar(x + w, [phases.get(k, {}).get("rmse_wz", 0.0) * 10.0 for k in ph_names], width=w, color=C_REAL, label="RMSE wz ×10 (rad/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(ph_labels)
    ax.set_ylabel("Error RMSE")
    ax.set_title(f"{disp} ({legend_lbl}) — Desglose de Error por Fase de Curva")
    ax.legend()
    save_fig(fig, out_dir / "error_by_corner_phase.png")
