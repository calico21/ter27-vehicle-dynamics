#!/usr/bin/env python3
"""
Unit & Integration Verification Suite for the TeR27 (280.0 kg) Limit Study
and MirenaSim (Godot 4 3D) Independent Physics Audit.

Certifies:
  1. TeR27 280.0 kg config consistency (mass, sprung mass, inertias, TV config)
  2. Zero TeR26 modification and untouched grip/aero constants
  3. All FSAE EV safety invariants across upstream 280 kg reports & setups
  4. MirenaSim Godot 4 3D independent audit parity + 5x 300-DPI pure-ASCII plots
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO_ROOT / "outputs" / "ter27_280kg_limit_study"
VEHICLES_DIR = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Config" / "vehicles"


def _load_ascii_json(path: Path) -> dict:
    assert path.exists(), f"Missing artifact: {path}"
    raw = path.read_text(encoding="utf-8")
    assert all(ord(ch) < 128 for ch in raw), f"Non-ASCII character found in {path}"
    return json.loads(raw)


def test_ter27_280kg_config_consistency_and_ter26_untouched() -> None:
    t27_veh = json.loads((VEHICLES_DIR / "ter27_4wd" / "vehicle.json").read_text(encoding="utf-8"))
    assert abs(float(t27_veh["mass"]) - 280.0) < 1e-6
    assert abs(float(t27_veh["sprung_mass"]) - 226.0) < 1e-6
    assert abs(float(t27_veh["unsprung_mass_total"]) - 54.0) < 1e-6
    assert abs(float(t27_veh["inertia"]["ixx"]) - 14.75) < 1e-3
    assert abs(float(t27_veh["inertia"]["iyy"]) - 80.63) < 1e-3
    assert abs(float(t27_veh["inertia"]["izz"]) - 79.58) < 1e-3
    assert abs(float(t27_veh["inertia"]["ixz"]) - (-6.13)) < 1e-3

    t27_tv = json.loads((VEHICLES_DIR / "ter27_4wd" / "tv_4wd_config.json").read_text(encoding="utf-8"))
    phys_sec = t27_tv.get("physical", t27_tv.get("vehicle", t27_tv))
    assert abs(float(phys_sec.get("mass_kg", 0.0)) - 280.0) < 1e-6

    # TeR26 RWD must remain untouched (distinct vehicle_id and mass != 280.0)
    t26_veh = json.loads((VEHICLES_DIR / "ter26_rwd" / "vehicle.json").read_text(encoding="utf-8"))
    assert t26_veh["vehicle_id"] == "ter26_rwd"
    assert abs(float(t26_veh["mass"]) - 280.0) > 1.0


def test_mass_and_powertrain_280kg_invariants() -> None:
    d = _load_ascii_json(STUDY_DIR / "mass_and_powertrain_report.json")
    assert abs(float(d["mass_update_280kg"]["mass_kg_new"]) - 280.0) < 1e-6
    assert d["mass_update_280kg"]["ter26_untouched"] is True
    assert abs(float(d["metadata"]["lambda_muy_fsg_rubbered"]) - 0.57) < 1e-9
    assert abs(float(d["metadata"]["lambda_muy_cold_test_oct"]) - 0.47) < 1e-9

    inv = d["invariants"]
    assert int(inv["hv_dc_violations"]) == 0
    assert int(inv["bspd_ev23_violations"]) == 0
    assert int(inv["skidpad_cones_hit"]) == 0
    assert float(inv["skidpad_min_clear_m"]) >= 0.150
    assert inv["grip_profiles_intact"] is True

    # Pure -10 kg mass reduction improves all 4 events
    pm = d["pure_mass_reduction_10kg_gains"]
    assert float(pm["skidpad_delta_ms"]) < 0.0
    assert float(pm["accel_75m_delta_ms"]) < 0.0
    assert float(pm["autocross_1184m_delta_ms"]) < 0.0
    assert float(pm["endurance_22km_delta_s"]) < 0.0


def test_chassis_aero_coopt_280kg_invariants() -> None:
    d = _load_ascii_json(STUDY_DIR / "chassis_aero_coopt_report.json")
    assert d["all_constraints_passed"] is True
    assert d["gradient_verification"]["all_passed_1e_6"] is True
    assert float(d["gradient_verification"]["worst_max_rel_err"]) < 1e-6

    meta = d["metadata"]
    assert abs(float(meta["fsg_rubbered_lambda_muy"]) - 0.57) < 1e-9
    assert abs(float(meta["cold_test_oct_lambda_muy"]) - 0.47) < 1e-9
    assert abs(float(meta["cfd_cl_a_m2"]) - 4.68) < 1e-6

    for ev_k in ("skidpad", "acceleration", "autocross"):
        ev = d["events"][ev_k]
        assert ev["constraints_passed"] is True
        assert float(ev["time_gains_ms"]["chassis_aero_isolated_ms"]) > 0.0

    for fn in ("chassis_aero_skidpad.json", "chassis_aero_accel.json", "chassis_aero_autocross.json"):
        sd = _load_ascii_json(STUDY_DIR / "setups" / fn)
        assert sd["vehicle_id"] == "ter27_4wd"
        assert abs(float(sd["mass_kg"]) - 280.0) < 1e-6


def test_trajectory_control_and_tube_mpc_280kg_invariants() -> None:
    d = _load_ascii_json(STUDY_DIR / "trajectory_and_control_report.json")
    assert d["all_constraints_passed"] is True
    assert d["adjoint_fd_verification"]["all_passed_1e_6"] is True
    assert float(d["adjoint_fd_verification"]["worst_rel_err"]) < 1e-6
    assert d["tube_mpc_safety_verification"]["all_tube_safety_passed"] is True

    assert abs(float(d["lambda_muy_fixed"]) - 0.57) < 1e-9
    assert abs(float(d["cold_test_oct_fixed"]) - 0.47) < 1e-9
    assert abs(float(d["aero_cla_fixed"]) - 4.68) < 1e-6
    assert abs(float(d["aero_cda_fixed"]) - 1.50) < 1e-6

    for ev_k in ("skidpad", "acceleration", "autocross"):
        ev = d["events"][ev_k]
        assert ev["constraints_passed"] is True
        base_t = float(ev["metrics_290kg_baseline"]["event_time_s"])
        opt_m = ev["metrics_280kg_limit_opt"]
        assert float(opt_m["event_time_s"]) < base_t
        assert int(opt_m["hv_violations"]) == 0
        assert int(opt_m["bspd_violations"]) == 0
        assert int(opt_m["cones_hit"]) == 0
        assert float(opt_m["max_dmz_dbeta_nm_rad"]) < 0.0
        assert float(opt_m["min_rh_f_mm"]) >= 10.0
        if ev_k in ("skidpad", "autocross"):
            assert float(opt_m["min_clear_m"]) >= 0.150

    tube_drop = d["tube_mpc_safety_verification"]["cases"]["skidpad_280kg_grip_drop_125ms_tube"]
    assert tube_drop["cones_hit_zero"] is True
    assert float(tube_drop["min_clear_m"]) >= 0.150
    assert float(tube_drop["max_dmz_dbeta_nm_rad"]) < 0.0


def test_thermal_and_endurance_22km_280kg_invariants() -> None:
    d = _load_ascii_json(STUDY_DIR / "thermal_and_endurance_report.json")
    inv = d["invariants"]
    assert inv["grip_profiles_intact"] is True
    assert inv["peak_tire_mu_th_le_1p0"] is True
    assert inv["part_a_zero_hv_and_bspd_violations"] is True
    assert inv["part_a_skidpad_min_clear_ge_0p150m"] is True
    assert inv["part_a_yaw_stable"] is True
    assert inv["part_b_coupled_280kg_within_3p50_kwh"] is True
    assert inv["part_b_coupled_280kg_zero_hv_and_bspd"] is True
    assert inv["part_b_coupled_280kg_zero_thermal_derating"] is True

    opt280 = d["part_b_endurance_22km"]["optimal_combined_280kg"]
    opt290 = d["part_b_endurance_22km"]["optimal_combined_290kg"]
    c280 = opt280["coupled_22km_19laps"]
    assert float(c280["e_net_22km_coupled_kwh"]) <= 3.50
    assert int(c280["motor_derating_events"]) == 0
    assert int(c280["inv_derating_events"]) == 0
    assert int(c280["hv_violations"]) == 0
    assert int(c280["bspd_violations"]) == 0
    assert float(c280["min_clear_m"]) >= 0.150
    assert float(opt280["score"]["pts_endurance_plus_efficiency"]) > float(
        opt290["score"]["pts_endurance_plus_efficiency"]
    )


def test_exported_280kg_limit_setups_and_mirenasim_audit_parity() -> None:
    setup_dir = STUDY_DIR / "setups"
    for fn in (
        "limit_280kg_skidpad.json",
        "limit_280kg_accel.json",
        "limit_280kg_autocross.json",
        "limit_280kg_endurance.json",
    ):
        sd = _load_ascii_json(setup_dir / fn)
        assert sd["vehicle_id"] == "ter27_4wd"
        assert abs(float(sd["mass_kg"]) - 280.0) < 1e-6

    audit = _load_ascii_json(STUDY_DIR / "mirenasim_audit_report.json")
    assert audit["all_limit_invariants_passed"] is True

    evs = audit["events_godot4_audit"]
    for ev_k in ("skidpad", "accel", "autocross", "endurance"):
        o = evs[ev_k]["godot4_280kg_limit"]
        inv = evs[ev_k]["invariants_280kg_limit"]
        assert bool(o["completed"]) is True
        assert abs(float(o["mass_kg"]) - 280.0) < 1e-6
        assert inv["all_invariants_passed"] is True
        assert int(o["cones_hit"]) == 0
        assert float(o["min_cone_clear_m"]) >= 0.150
        assert int(o["bumpstop_hits"]) == 0
        assert float(o["min_rh_f_mm"]) >= 10.0
        assert int(o["hv_violations"]) == 0
        assert int(o["bspd_violations"]) == 0
        assert float(o["dmz_dbeta_nm_rad"]) < 0.0

    # 280 kg limit faster than 290 kg baseline in Godot 4 on Skidpad, Accel, Autocross
    for ev_k in ("skidpad", "accel", "autocross"):
        assert float(evs[ev_k]["godot4_delta_time_s"]) < 0.0

    # Sim-to-Sim parity on Acceleration (75 m) within 3.0%
    assert abs(float(evs["accel"]["sim2sim_parity_delta_pct"])) < 3.0

    # Skidpad under SENSOR_TRACK_REALISM still passes all invariants
    skid_noisy = audit["auxiliary_godot4_checks"]["skidpad_280kg_sensor_realism"]
    assert skid_noisy["invariants"]["all_invariants_passed"] is True

    # Verify all 5 300-DPI pure-ASCII plots exist and are non-trivial
    expected_figs = [
        "fig01_ter27_290kg_vs_280kg_waterfall_attribution.png",
        "fig02_powertrain_gear_ratio_and_loss_recovery.png",
        "fig03_free_trajectory_and_cone_margin_audit.png",
        "fig04_thermal_window_and_endurance_22km_pareto.png",
        "fig05_mirenasim_godot4_280kg_audit_parity.png",
    ]
    for fn in expected_figs:
        fp = STUDY_DIR / fn
        assert fp.exists(), f"Missing plot: {fp}"
        assert fp.stat().st_size > 30_000, f"Plot too small: {fp}"
        with Image.open(fp) as im:
            dpi = im.info.get("dpi", (0, 0))
            assert dpi[0] >= 299 and dpi[1] >= 299, f"Plot {fn} DPI={dpi} < 300"

    # Executive SUMMARY_REPORT.json
    summary = _load_ascii_json(STUDY_DIR / "SUMMARY_REPORT.json")
    assert summary["all_gates_passed"] is True
    assert all(summary["upstream_reports_present"].values())
