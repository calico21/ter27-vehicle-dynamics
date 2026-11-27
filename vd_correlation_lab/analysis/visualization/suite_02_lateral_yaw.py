#!/usr/bin/env python3
"""Suite 02: Dinámica lateral y guiñada (ganancia, Kus, scatter ±1σ, g-g y ay vs vx*wz)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .style import C_ACCENT, C_HORIZON, C_OPEN, C_REAL, save_fig


def generate(out_dir: Path, runs: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_lbl = payload.get("legend_label", "sim_TeR")
    disp = payload.get("display_name", "TeR")
    wb = float(payload.get("wheelbase_m", 1.55))

    track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
    vx_r = np.concatenate([r["real"]["vx_ms"].to_numpy() for r in track_runs])
    ax_r = np.concatenate([r["real"]["ax_ms2"].to_numpy() for r in track_runs])
    ay_r = np.concatenate([r["real"]["ay_ms2"].to_numpy() for r in track_runs])
    wz_r = np.concatenate([r["real"]["wz_rads"].to_numpy() for r in track_runs])
    str_r = np.concatenate([r["real"]["steer_deg"].to_numpy() for r in track_runs])

    vx_s = np.concatenate([r["sim_open"]["vx_sim"].to_numpy() for r in track_runs])
    ax_s = np.concatenate([r["sim_open"]["ax_sim"].to_numpy() for r in track_runs])
    ay_s = np.concatenate([r["sim_open"]["ay_sim"].to_numpy() for r in track_runs])
    wz_s = np.concatenate([r["sim_open"]["wz_sim"].to_numpy() for r in track_runs])

    # 1. ganancia de guiñada (wz / delta) vs vx y diagrama de subviraje kus
    m_lat = (vx_r > 4.0) & (np.abs(str_r) > 1.5) & (np.abs(ay_r) > 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if np.sum(m_lat) > 30:
        str_rad = np.radians(str_r[m_lat])
        gain_r = wz_r[m_lat] / str_rad
        gain_s = wz_s[m_lat] / str_rad
        ok_g = (np.abs(gain_r) < 10.0) & (np.abs(gain_s) < 10.0)
        axes[0].scatter(vx_r[m_lat][ok_g], gain_r[ok_g], s=4, alpha=0.25, color=C_REAL, label="Real CAN")
        axes[0].scatter(vx_s[m_lat][ok_g], gain_s[ok_g], s=4, alpha=0.25, color=C_HORIZON, label=legend_lbl)
        axes[0].set_xlabel("Velocidad vx (m/s)")
        axes[0].set_ylabel("Ganancia de guiñada wz / δ (1/s)")
        axes[0].set_title("Ganancia de Guiñada frente a Velocidad")
        axes[0].legend()

        ay_g_r = ay_r[m_lat] / 9.80665
        ay_g_s = ay_s[m_lat] / 9.80665
        us_r = str_r[m_lat] - np.degrees(wb * wz_r[m_lat] / vx_r[m_lat])
        us_s = str_r[m_lat] - np.degrees(wb * wz_s[m_lat] / np.maximum(vx_s[m_lat], 4.0))
        axes[1].scatter(ay_g_r, us_r, s=4, alpha=0.22, color=C_REAL, label=f"Real (Kus={payload.get('kus_real_deg_g', 0.0):.2f} °/g)")
        axes[1].scatter(ay_g_s, us_s, s=4, alpha=0.22, color=C_HORIZON, label=f"{legend_lbl} (Kus={payload.get('kus_sim_deg_g', 0.0):.2f} °/g)")
        axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
        axes[1].set_xlabel("Aceleración lateral ay (g)")
        axes[1].set_ylabel("δ_real - δ_Ackermann (°)")
        axes[1].set_title("Diagrama de Subviraje (Kus)")
        axes[1].legend()
    fig.suptitle(f"{disp} — Respuesta Direccional y Subviraje", fontweight="bold")
    save_fig(fig, out_dir / "yaw_gain_and_understeer.png")

    # 2. scatter wz y ay (real vs simulado) con bandas ±1σ y recta y=x
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, rv, sv, title, unit in [
        (axes[0], ay_r, ay_s, "Aceleración Lateral ay", "m/s²"),
        (axes[1], wz_r, wz_s, "Velocidad de Guiñada wz", "rad/s"),
    ]:
        ok = np.isfinite(rv) & np.isfinite(sv) & (vx_r > 3.0)
        r_v, s_v = rv[ok], sv[ok]
        lo = min(float(np.percentile(r_v, 1)), float(np.percentile(s_v, 1)))
        hi = max(float(np.percentile(r_v, 99)), float(np.percentile(s_v, 99)))
        sig = float(np.std(s_v - r_v))
        xx = np.linspace(lo, hi, 100)
        ax.scatter(r_v, s_v, s=4, alpha=0.2, color=C_HORIZON)
        ax.plot(xx, xx, color=C_REAL, lw=1.4, label="y = x")
        ax.fill_between(xx, xx - sig, xx + sig, color=C_OPEN, alpha=0.14, label=f"±1σ ({sig:.2f} {unit})")
        ax.set_xlabel(f"Real ({unit})")
        ax.set_ylabel(f"{legend_lbl} ({unit})")
        ax.set_title(title)
        ax.legend()
    fig.suptitle(f"{disp} — Dispersión Real vs {legend_lbl}", fontweight="bold")
    save_fig(fig, out_dir / "scatter_ay_wz_1sigma.png")

    # 3. diagrama g-g coloreado x vel
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    m_v = vx_r > 3.0
    sc0 = axes[0].scatter(ay_r[m_v] / 9.80665, ax_r[m_v] / 9.80665, c=vx_r[m_v], cmap="viridis", s=5, alpha=0.45)
    axes[0].set_title("Envolvente g-g Real CAN")
    axes[0].set_xlabel("ay (g)")
    axes[0].set_ylabel("ax (g)")
    sc1 = axes[1].scatter(ay_s[m_v] / 9.80665, ax_s[m_v] / 9.80665, c=vx_s[m_v], cmap="viridis", s=5, alpha=0.45)
    axes[1].set_title(f"Envolvente g-g {legend_lbl}")
    axes[1].set_xlabel("ay (g)")
    fig.colorbar(sc1, ax=axes.ravel().tolist(), label="Velocidad vx (m/s)", shrink=0.85)
    fig.suptitle(f"{disp} — Diagrama g-g (Círculo de Fricción)", fontweight="bold")
    save_fig(fig, out_dir / "gg_envelope.png")

    # 4. consistencia cinemática lateral (ay medido vs vx * wz)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cent_r = vx_r[m_v] * wz_r[m_v]
    cent_s = vx_s[m_v] * wz_s[m_v]
    ax.scatter(cent_r, ay_r[m_v], s=4, alpha=0.25, color=C_REAL, label="Real: ay vs vx·wz")
    ax.scatter(cent_s, ay_s[m_v], s=4, alpha=0.25, color=C_HORIZON, label=f"{legend_lbl}: ay vs vx·wz")
    lim = max(12.0, float(np.percentile(np.abs(ay_r[m_v]), 99)))
    ax.plot([-lim, lim], [-lim, lim], color=C_ACCENT, lw=1.2, ls="--", label="ay = vx·wz (deriva nula)")
    ax.set_xlabel("Término centrípeto vx·wz (m/s²)")
    ax.set_ylabel("Aceleración lateral ay (m/s²)")
    ax.set_title(f"{disp} — Consistencia Cinemática Lateral (Auditoría β̇)")
    ax.legend()
    save_fig(fig, out_dir / "kinematic_ay_vs_vx_wz.png")
