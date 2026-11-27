#!/usr/bin/env python3
"""Suite 06: Diagnóstico 4WD y Torque Vectoring (pares 4 ruedas, Mz activo y ASR/TC)."""

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

    track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
    df_s = track_runs[0]["sim_open"]
    df_r = track_runs[0]["real"]
    t = df_s["t_s"].to_numpy()
    sl = slice(0, min(len(t), 3500))

    # 1. par individual en las 4 ruedas y diferencia izq-der
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    t_fl = df_s["torque_fl"].to_numpy()
    t_fr = df_s["torque_fr"].to_numpy()
    t_rl = df_s["torque_rl"].to_numpy()
    t_rr = df_s["torque_rr"].to_numpy()

    axes[0].plot(t[sl], t_fl[sl], color=C_FL, lw=1.0, label="T_FL")
    axes[0].plot(t[sl], t_fr[sl], color=C_FR, lw=1.0, label="T_FR")
    axes[0].plot(t[sl], t_rl[sl], color=C_RL, lw=1.0, label="T_RL")
    axes[0].plot(t[sl], t_rr[sl], color=C_RR, lw=1.0, label="T_RR")
    axes[0].set_ylabel("Par rueda / inversor (Nm)")
    axes[0].set_title("Reparto Individual de Par en las 4 Ruedas")
    axes[0].legend(ncol=4)

    axes[1].plot(t[sl], (t_fr - t_fl)[sl], color=C_HORIZON, lw=1.1, label="ΔT Delantero (FR - FL)")
    axes[1].plot(t[sl], (t_rr - t_rl)[sl], color=C_OPEN, lw=1.1, label="ΔT Trasero (RR - RL)")
    axes[1].axhline(0.0, color="k", lw=0.7)
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Diferencia izq-der ΔT (Nm)")
    axes[1].set_title("Diferencial Electrónico Izquierda - Derecha")
    axes[1].legend()
    fig.suptitle(f"{disp} ({legend_lbl}) — Control de Par 4 Ruedas", fontweight="bold")
    save_fig(fig, out_dir / "wheel_torques_and_split.png")

    # 2. momento de guiñada activo δmz frente a error de guiñada y ángulo de volante
    mz = df_s["tv_mz_nm"].to_numpy()
    wz_err = df_s["tv_wz_ref_rads"].to_numpy() - df_s["wz_sim"].to_numpy()
    str_deg = df_r["steer_deg"].to_numpy()
    m_v = df_s["vx_sim"].to_numpy() > 3.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if np.any(m_v):
        axes[0].scatter(wz_err[m_v], mz[m_v], s=5, alpha=0.3, color=C_HORIZON)
        axes[1].scatter(str_deg[m_v], mz[m_v], s=5, alpha=0.3, color=C_OPEN)
    axes[0].set_xlabel("Error de guiñada (wz_ref - wz) (rad/s)")
    axes[0].set_ylabel("Momento activo ΔMz (Nm)")
    axes[0].set_title("ΔMz frente a Error de Guiñada")
    axes[1].set_xlabel("Ángulo de volante δ (°)")
    axes[1].set_ylabel("Momento activo ΔMz (Nm)")
    axes[1].set_title("ΔMz frente a Ángulo de Dirección")
    fig.suptitle(f"{disp} ({legend_lbl}) — Momento de Guiñada Activo (TV)", fontweight="bold")
    save_fig(fig, out_dir / "active_mz_response.png")

    # 3. intervención de control de tracción (asr/tc) x rueda
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    tc_act = df_s["tc_active"].to_numpy()
    axes[0].fill_between(t[sl], 0, tc_act[sl], color=C_OPEN, alpha=0.4, label="TC / ASR Activo")
    axes[0].set_ylim(-0.05, 1.2)
    axes[0].set_ylabel("Flag TC")
    axes[0].set_title("Activación del Control de Tracción (ASR / TC)")
    axes[0].legend()

    for col, lbl, c in [("fl", "FL", C_FL), ("fr", "FR", C_FR), ("rl", "RL", C_RL), ("rr", "RR", C_RR)]:
        axes[1].plot(t[sl], np.clip(df_s[f"kappa_{col}"].to_numpy()[sl], -0.25, 0.35), lw=0.9, color=c, label=f"κ {lbl}")
    axes[1].axhline(0.11, color=C_REAL, ls="--", lw=0.9, label="Umbral óptimo κ≈0.11")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Slip ratio κ (-)")
    axes[1].legend(ncol=5)
    fig.suptitle(f"{disp} ({legend_lbl}) — Intervención ASR / TC", fontweight="bold")
    save_fig(fig, out_dir / "asr_tc_intervention.png")
