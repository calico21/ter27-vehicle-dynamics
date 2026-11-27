#!/usr/bin/env python3
"""
CLI unificado del laboratorio de correlación multi-vehículo y multi-modelo.

Uso:
  python3 vd_correlation_lab/run_lab.py --vehicle ter26_rwd --model full_14dof
  python3 vd_correlation_lab/run_lab.py --vehicle ter26_rwd --model all
  python3 vd_correlation_lab/run_lab.py --vehicle all --model all
  python3 vd_correlation_lab/run_lab.py --compare ter26_rwd ter27_4wd --model fast_3dof
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from vd_correlation_lab.analysis.benchmarks.godot_runner import ensure_synthetic_log, run_godot_batch
from vd_correlation_lab.control.sensors_ekf.log_parser import DbcDatabase, load_log_to_grid
from vd_correlation_lab.analysis.benchmarks.metrics_engine import compute_corner_phase_errors, compute_kus, compute_signal_metrics
from vd_correlation_lab.analysis.benchmarks.report_generator import write_vehicle_results
from vd_correlation_lab.control.sensors_ekf.signal_conditioning import condition_telemetry
from plotting import (
    style,
    suite_01_global_dynamics,
    suite_02_lateral_yaw,
    suite_03_longitudinal_power,
    suite_04_suspension_aero,
    suite_05_tires_envelope,
    suite_06_4wd_torque_vec,
    suite_07_error_regimes,
    suite_08_frequency_spectral,
    suite_09_cross_vehicle,
    suite_10_cross_model,
)

AVAILABLE_MODELS = ["full_14dof", "fast_3dof"]

SIGNALS_TO_EVAL = [
    ("vx_ms", "vx_sim"),
    ("ax_ms2", "ax_sim"),
    ("ay_ms2", "ay_sim"),
    ("wz_rads", "wz_sim"),
    ("wheel_rpm_fl", "wheel_rpm_fl_sim"),
    ("wheel_rpm_fr", "wheel_rpm_fr_sim"),
    ("wheel_rpm_rl", "wheel_rpm_rl_sim"),
    ("wheel_rpm_rr", "wheel_rpm_rr_sim"),
]


def _load_vehicle_config_meta(vehicle_id: str) -> Dict[str, Any]:
    vjson = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Config" / "vehicles" / vehicle_id / "vehicle.json"
    if vjson.exists():
        with open(vjson, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _prepare_vehicle_logs(vehicle_id: str, dbc: DbcDatabase) -> Dict[str, Any]:
    vdir = LAB_DIR / "vehicles" / vehicle_id
    calib_path = vdir / "sensor_calibration.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Falta sensor_calibration.json en {vdir}")
    with open(calib_path, "r", encoding="utf-8") as fh:
        calib = json.load(fh)

    vmeta = _load_vehicle_config_meta(vehicle_id)
    wb = float(vmeta.get("wheelbase", 1.55))
    logs_dir = vdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.csv")))
    if not log_files and bool(calib.get("is_synthetic", False)):
        print(f"[{vehicle_id}] Sin logs CAN de pista; generando maniobra sintética de referencia...")
        log_files = [ensure_synthetic_log(REPO_ROOT, logs_dir, vehicle_id, calib)]
    if not log_files:
        raise FileNotFoundError(f"No hay logs (.log o .csv) en {logs_dir}")

    print(f"[{vehicle_id}] Acondicionando {len(log_files)} log(s)...")
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"vd_lab_{vehicle_id}_"))
    base_runs: List[Dict[str, Any]] = []
    for lf in log_files:
        df_raw, raw_streams, present_counts, is_pre = load_log_to_grid(lf, dbc, fs_hz=100.0)
        df_cond, meta = condition_telemetry(df_raw, raw_streams, present_counts, calib, is_preconditioned_csv=is_pre)
        in_csv = tmp_dir / f"{lf.stem}_cond.csv"
        df_cond.to_csv(in_csv, index=False)
        base_runs.append({
            "name": lf.stem,
            "in_csv": in_csv,
            "real": df_cond,
            "meta": meta,
            "duration_s": float(df_cond["t_s"].iloc[-1] - df_cond["t_s"].iloc[0]) if len(df_cond) > 1 else 0.0,
        })

    return {"vdir": vdir, "calib": calib, "wb": wb, "tmp_dir": tmp_dir, "base_runs": base_runs}


def evaluate_vehicle_model(vehicle_id: str, prep: Dict[str, Any], model_id: str, no_plots: bool = False) -> Dict[str, Any]:
    vdir: Path = prep["vdir"]
    calib: Dict[str, Any] = prep["calib"]
    wb: float = prep["wb"]
    tmp_dir: Path = prep["tmp_dir"]
    ecu_mode = int(calib.get("ecu_mode", 0))
    tv_map_id = int(calib.get("tv_map_id", 2))

    runs: List[Dict[str, Any]] = []
    batch_jobs: List[Dict[str, Any]] = []
    for br in prep["base_runs"]:
        out_h = tmp_dir / f"{br['name']}_{model_id}_horizon10s.csv"
        out_o = tmp_dir / f"{br['name']}_{model_id}_openloop.csv"
        batch_jobs.append({
            "kind": "replay", "vehicle_id": vehicle_id, "setup": "default", "fidelity": model_id,
            "input_csv": str(br["in_csv"]), "out_csv": str(out_h), "horizon_s": 10.0,
            "ecu_mode": ecu_mode, "tv_map_id": tv_map_id,
        })
        batch_jobs.append({
            "kind": "replay", "vehicle_id": vehicle_id, "setup": "default", "fidelity": model_id,
            "input_csv": str(br["in_csv"]), "out_csv": str(out_o), "horizon_s": 0.0,
            "ecu_mode": ecu_mode, "tv_map_id": tv_map_id,
        })
        runs.append({
            "name": br["name"], "real": br["real"], "meta": br["meta"],
            "duration_s": br["duration_s"], "out_h": out_h, "out_o": out_o,
        })

    print(f"[{vehicle_id}][{model_id}] Ejecutando {len(batch_jobs)} simulaciones en Godot headless...")
    t0 = time.perf_counter()
    run_godot_batch(REPO_ROOT, batch_jobs, timeout_s=240)
    elapsed_s = time.perf_counter() - t0

    for r in runs:
        r["sim_horizon"] = pd.read_csv(r["out_h"])
        r["sim_open"] = pd.read_csv(r["out_o"])
        active = r["real"]["is_active"].to_numpy()
        r["metrics_horizon"] = {}
        r["metrics_open"] = {}
        for cr, cs in SIGNALS_TO_EVAL:
            if cr in r["real"].columns and cs in r["sim_horizon"].columns:
                r["metrics_horizon"][cr] = compute_signal_metrics(r["real"][cr].to_numpy()[active], r["sim_horizon"][cs].to_numpy()[active])
                r["metrics_open"][cr] = compute_signal_metrics(r["real"][cr].to_numpy()[active], r["sim_open"][cs].to_numpy()[active])

    track_runs = [r for r in runs if r["meta"].get("log_type", "track") == "track"] or runs
    paddock_count = sum(1 for r in runs if r["meta"].get("log_type") == "paddock_slow")

    pooled_h: Dict[str, Any] = {}
    pooled_o: Dict[str, Any] = {}
    for cr, cs in SIGNALS_TO_EVAL:
        if all(cr in r["real"].columns for r in track_runs):
            real_cat = np.concatenate([r["real"][cr].to_numpy()[r["real"]["is_active"].to_numpy()] for r in track_runs])
            sim_h_cat = np.concatenate([r["sim_horizon"][cs].to_numpy()[r["real"]["is_active"].to_numpy()] for r in track_runs])
            sim_o_cat = np.concatenate([r["sim_open"][cs].to_numpy()[r["real"]["is_active"].to_numpy()] for r in track_runs])
            pooled_h[cr] = compute_signal_metrics(real_cat, sim_h_cat)
            pooled_o[cr] = compute_signal_metrics(real_cat, sim_o_cat)

    vx_cat = np.concatenate([r["real"]["vx_ms"].to_numpy() for r in track_runs])
    ax_cat = np.concatenate([r["real"]["ax_ms2"].to_numpy() for r in track_runs])
    ay_cat = np.concatenate([r["real"]["ay_ms2"].to_numpy() for r in track_runs])
    wz_cat = np.concatenate([r["real"]["wz_rads"].to_numpy() for r in track_runs])
    str_cat = np.concatenate([r["real"]["steer_deg"].to_numpy() for r in track_runs])

    vx_s_cat = np.concatenate([r["sim_open"]["vx_sim"].to_numpy() for r in track_runs])
    ax_s_cat = np.concatenate([r["sim_open"]["ax_sim"].to_numpy() for r in track_runs])
    ay_s_cat = np.concatenate([r["sim_open"]["ay_sim"].to_numpy() for r in track_runs])
    wz_s_cat = np.concatenate([r["sim_open"]["wz_sim"].to_numpy() for r in track_runs])

    kus_real, _ = compute_kus(vx_cat, ay_cat, wz_cat, str_cat, wheelbase_m=wb)
    kus_sim, _ = compute_kus(vx_s_cat, ay_s_cat, wz_s_cat, str_cat, wheelbase_m=wb)
    corner_phases = compute_corner_phase_errors(vx_cat, ax_cat, ay_cat, wz_cat, ax_s_cat, ay_s_cat, wz_s_cat)

    base_lbl = calib.get("legend_label", f"sim_{vehicle_id}")
    payload: Dict[str, Any] = {
        "vehicle_id": vehicle_id,
        "model_id": model_id,
        "display_name": calib.get("display_name", vehicle_id),
        "legend_label": f"{base_lbl} ({model_id})",
        "is_synthetic": bool(calib.get("is_synthetic", False)),
        "elapsed_s": elapsed_s,
        "wheelbase_m": wb,
        "wheel_radius_m": float(calib.get("wheel_radius_m", 0.2032)),
        "track_runs_count": len([r for r in runs if r["meta"].get("log_type", "track") == "track"]),
        "paddock_runs_count": paddock_count,
        "pooled_horizon": pooled_h,
        "pooled_open": pooled_o,
        "kus_real_deg_g": kus_real,
        "kus_sim_deg_g": kus_sim,
        "corner_phases_open": corner_phases,
        "baseline_reference": calib.get("baseline_reference", {}),
        "runs": [
            {
                "name": r["name"], "duration_s": r["duration_s"], "meta": r["meta"],
                "metrics_horizon": r["metrics_horizon"], "metrics_open": r["metrics_open"],
            }
            for r in runs
        ],
    }

    model_dir = vdir / "models" / model_id
    write_vehicle_results(model_dir / "results", payload)

    plots_dir = model_dir / "plots"
    if not no_plots:
        print(f"[{vehicle_id}][{model_id}] Generando 8 suites a 300 DPI en {plots_dir}...")
        suite_01_global_dynamics.generate(plots_dir / "01_dinamica_global", runs, payload)
        suite_02_lateral_yaw.generate(plots_dir / "02_lateral_y_guiniada", runs, payload)
        suite_03_longitudinal_power.generate(plots_dir / "03_longitudinal_y_ruedas", runs, payload)
        suite_04_suspension_aero.generate(plots_dir / "04_suspension_y_aero", runs, payload)
        suite_05_tires_envelope.generate(plots_dir / "05_neumaticos_y_envolvente", runs, payload)
        suite_06_4wd_torque_vec.generate(plots_dir / "06_4wd_y_torque_vectoring", runs, payload)
        suite_07_error_regimes.generate(plots_dir / "07_analisis_error_y_regimenes", runs, payload)
        suite_08_frequency_spectral.generate(plots_dir / "08_espectro_frecuencia", runs, payload)

    return {"payload": payload, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Laboratorio de correlación multi-vehículo y multi-modelo Tecnun eRacing")
    parser.add_argument("--vehicle", type=str, default=None, help="ter26_rwd | ter27_4wd | all")
    parser.add_argument("--model", type=str, default="all", help="full_14dof | fast_3dof | all")
    parser.add_argument("--compare", nargs="+", default=None, help="Lista de vehículos a comparar (ej. ter26_rwd ter27_4wd)")
    parser.add_argument("--no-plots", action="store_true", help="Omitir generación de gráficas 300 DPI")
    args = parser.parse_args()

    style.apply_style()
    dbc = DbcDatabase(LAB_DIR / "dbc" / "TER.dbc")

    available_vehicles = sorted([p.name for p in (LAB_DIR / "vehicles").iterdir() if p.is_dir() and (p / "sensor_calibration.json").exists()])
    if args.compare:
        target_vehicles = args.compare
    elif args.vehicle == "all" or args.vehicle is None:
        target_vehicles = available_vehicles
    else:
        target_vehicles = [args.vehicle]

    target_models = AVAILABLE_MODELS if args.model == "all" else [args.model]

    by_model: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in target_models}
    for vid in target_vehicles:
        prep = _prepare_vehicle_logs(vid, dbc)
        by_vehicle_models: Dict[str, Dict[str, Any]] = {}
        for mid in target_models:
            res = evaluate_vehicle_model(vid, prep, mid, no_plots=args.no_plots)
            by_vehicle_models[mid] = res
            by_model[mid][vid] = res
            p = res["payload"]
            ph, po = p["pooled_horizon"], p["pooled_open"]
            print(
                f"[{vid}][{mid}] COMPLETADO ({p['elapsed_s']:.1f}s) | "
                f"Horiz 10s r(vx={ph['vx_ms']['r']:.3f}, ay={ph['ay_ms2']['r']:.3f}, wz={ph['wz_rads']['r']:.3f}) | "
                f"Lazo Abierto r(vx={po['vx_ms']['r']:.3f}, ay={po['ay_ms2']['r']:.3f}, wz={po['wz_rads']['r']:.3f})"
            )

        if "full_14dof" in by_vehicle_models and "fast_3dof" in by_vehicle_models:
            cm_dir = LAB_DIR / "results" / "cross_model" / vid
            print(f"[CROSS-MODEL][{vid}] Generando comparativa 14-DOF vs 3-DOF en {cm_dir}...")
            suite_10_cross_model.generate_cross_model(cm_dir, vid, prep["calib"].get("display_name", vid), by_vehicle_models)

    for mid, vdict in by_model.items():
        if len(vdict) >= 2:
            cv_dir = LAB_DIR / "results" / "cross_vehicle" / mid
            print(f"[CROSS-VEHICLE][{mid}] Generando comparativa entre coches en {cv_dir}...")
            suite_09_cross_vehicle.generate_cross_vehicle(cv_dir, vdict)


if __name__ == "__main__":
    main()
