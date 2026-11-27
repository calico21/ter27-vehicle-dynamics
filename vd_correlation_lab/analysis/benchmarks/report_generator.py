#!/usr/bin/env python3
"""
Generador automático de metrics.json e INFORME_CORRELACION.md por vehículo.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict


def _fmt(v: Any, dec: int = 4) -> str:
    try:
        fv = float(v)
        if fv != fv:
            return "N/A"
        return f"{fv:.{dec}f}"
    except (TypeError, ValueError):
        return "N/A"


def write_vehicle_results(results_dir: Path, payload: Dict[str, Any]) -> None:
    """Guarda metrics.json y genera INFORME_CORRELACION.md en vehicles/<id>/results/."""
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    vid = payload.get("vehicle_id", "unknown")
    disp = payload.get("display_name", vid)
    legend = payload.get("legend_label", f"sim_{disp}")
    is_synth = bool(payload.get("is_synthetic", False))
    tag = "Validación Sintética / Pre-pista" if is_synth else "Validación en Pista (Logs CAN Reales)"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Informe de Correlación Sim-to-Real — {disp} (`{vid}`)",
        "",
        f"- **Etiqueta de modelo:** `{legend}`",
        f"- **Origen de datos:** **{tag}**",
        f"- **Fecha de generación:** {now_str}",
        f"- **Tandas analizadas:** {len(payload.get('runs', []))} ({payload.get('track_runs_count', 0)} de pista, {payload.get('paddock_runs_count', 0)} de paddock lento)",
        "",
    ]

    if is_synth:
        lines.extend([
            "> **Nota:** Aún no existen logs CAN de pista para este vehículo; la evaluación se ha ejecutado sobre maniobras sintéticas de referencia con ruido instrumental realista para verificar el pipeline completo (incluida la suite `06_4wd_y_torque_vectoring`).",
            "",
        ])

    lines.extend([
        "## 1. Métricas Agrupadas (Tandas de Pista)",
        "",
        "| Señal | Horizonte 10 s (`r`) | Horizonte 10 s (`R²`) | Horizonte 10 s (`RMSE`) | Lazo Abierto (`r`) | Lazo Abierto (`R²`) | Lazo Abierto (`RMSE`) | Lag óptimo (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    pooled_h = payload.get("pooled_horizon", {})
    pooled_o = payload.get("pooled_open", {})
    sig_labels = [
        ("vx_ms", "Velocidad `vx` (m/s)"),
        ("ax_ms2", "Acel. longitudinal `ax` (m/s²)"),
        ("ay_ms2", "Acel. lateral `ay` (m/s²)"),
        ("wz_rads", "Guiñada `wz` (rad/s)"),
        ("wheel_rpm_rl", "RPM rueda RL"),
        ("wheel_rpm_rr", "RPM rueda RR"),
    ]
    for key, label in sig_labels:
        mh = pooled_h.get(key, {})
        mo = pooled_o.get(key, {})
        lines.append(
            f"| {label} | {_fmt(mh.get('r'))} | {_fmt(mh.get('r2'))} | {_fmt(mh.get('rmse'), 3)} | "
            f"{_fmt(mo.get('r'))} | {_fmt(mo.get('r2'))} | {_fmt(mo.get('rmse'), 3)} | {_fmt(mo.get('optimal_lag_ms'), 1)} |"
        )

    lines.extend([
        "",
        "## 2. Subviraje y Consistencia por Fase de Curva",
        "",
        f"- **Gradiente de subviraje `Kus` Real:** {_fmt(payload.get('kus_real_deg_g'), 2)} deg/g",
        f"- **Gradiente de subviraje `Kus` {legend}:** {_fmt(payload.get('kus_sim_deg_g'), 2)} deg/g",
        "",
        "| Fase de curva | Muestras | RMSE `ay` (m/s²) | RMSE `ax` (m/s²) | RMSE `wz` (rad/s) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    phases = payload.get("corner_phases_open", {})
    for pk, plab in [("entrada", "Entrada (Frenada + Turn-in)"), ("vertice", "Vértice (Apoyo)"), ("salida", "Salida (Tracción)")]:
        p = phases.get(pk, {})
        lines.append(
            f"| {plab} | {p.get('n', 0)} | {_fmt(p.get('rmse_ay'), 3)} | {_fmt(p.get('rmse_ax'), 3)} | {_fmt(p.get('rmse_wz'), 3)} |"
        )

    fid_cmp = payload.get("fidelity_comparison", {})
    if isinstance(fid_cmp, dict) and fid_cmp.get("full_14dof") and fid_cmp.get("fast_3dof"):
        r14 = fid_cmp["full_14dof"]
        r3 = fid_cmp["fast_3dof"]
        sp = float(fid_cmp.get("speedup", 1.0))
        lines.extend([
            "",
            "## 2b. Comparativa de Solvers (`full_14dof` vs `fast_3dof` a 200 Hz)",
            "",
            f"- **Tiempo batch `full_14dof`:** {_fmt(r14.get('elapsed_s'), 2)} s",
            f"- **Tiempo batch `fast_3dof (200 Hz)`:** {_fmt(r3.get('elapsed_s'), 2)} s (**Speedup: {_fmt(sp, 2)}×**)",
            "",
            "| Señal | Real CAN vs `full_14dof` (`r` H10s) | Real CAN vs `fast_3dof 200Hz` (`r` H10s) | Real CAN vs `full_14dof` (`r` Lazo Abierto) | Real CAN vs `fast_3dof 200Hz` (`r` Lazo Abierto) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for key, label in sig_labels[:4]:
            h14 = r14.get("pooled_horizon", {}).get(key, {}).get("r")
            h3 = r3.get("pooled_horizon", {}).get(key, {}).get("r")
            o14 = r14.get("pooled_open", {}).get(key, {}).get("r")
            o3 = r3.get("pooled_open", {}).get(key, {}).get("r")
            lines.append(f"| {label} | {_fmt(h14)} | {_fmt(h3)} | {_fmt(o14)} | {_fmt(o3)} |")

    lines.extend([
        "",
        "## 3. Desglose por Tanda",
        "",
        "| Tanda | Tipo | Duración (s) | EMI filtrados | Offset volante (°) | `r(ay)` Horiz. 10s | `r(wz)` Horiz. 10s | `r(vx)` Lazo Abierto |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in payload.get("runs", []):
        mh = r.get("metrics_horizon", {})
        mo = r.get("metrics_open", {})
        meta = r.get("meta", {})
        lines.append(
            f"| `{r.get('name', '')}` | `{meta.get('log_type', 'track')}` | {_fmt(r.get('duration_s'), 1)} | "
            f"{meta.get('emi_spikes_filtered', 0)} | {_fmt(meta.get('steer_zero_deg'), 2)} | "
            f"{_fmt(mh.get('ay_ms2', {}).get('r'))} | {_fmt(mh.get('wz_rads', {}).get('r'))} | {_fmt(mo.get('vx_ms', {}).get('r'))} |"
        )

    lines.extend([
        "",
        "## 4. Batería de Gráficas Generadas (300 DPI)",
        "",
        "Las gráficas se encuentran en `plots/` organizadas en las siguientes carpetas:",
        "- `01_dinamica_global/`: series temporales, resumen por tanda y evolución respecto a línea base.",
        "- `02_lateral_y_guiniada/`: ganancia de guiñada, subviraje `Kus`, scatter `±1σ`, diagrama `g-g` y consistencia `ay vs vx·wz`.",
        "- `03_longitudinal_y_ruedas/`: velocidades de rueda vs `vx`, slip ratios `kappa`, respuesta a pedales y ventanas de suelta/trail braking.",
        "- `04_suspension_y_aero/`: gradientes de balanceo/cabeceo, cargas `Fz`, reparto lateral `LLTD`, velocidades de amortiguador y downforce.",
        "- `05_neumaticos_y_envolvente/`: ángulos de deriva `alpha_f/alpha_r`, curvas de eje `Fy(alpha)` y utilización combinada de adherencia.",
        "- `06_4wd_y_torque_vectoring/`: par en las 4 ruedas, momento activo `Mz` vs error de guiñada e intervención de control de tracción `ASR/TC`.",
        "- `07_analisis_error_y_regimenes/`: mapas de calor 2D de error `(vx, |ay|)` y `(ax, ay)`, histogramas de residuos, barrido de lag y error por fase.",
        "- `08_espectro_frecuencia/`: PSD/FFT `0-50 Hz`, función de transferencia y coherencia espectral `γ²(f)`.",
        "",
    ])

    report_path = results_dir / "INFORME_CORRELACION.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
