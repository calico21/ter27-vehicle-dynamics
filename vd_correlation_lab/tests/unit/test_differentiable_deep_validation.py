#!/usr/bin/env python3
"""
Deep Validation & MirenaSim (Godot 4) Sim-to-Sim Parity Verification Suite.

Certifies all 5 workstreams in outputs/differentiable_deep_validation/:
  1. TeR26 empirical reverse-mode adjoint calibration & C2 smoothing audit
  2. TeR27 warm-start 14-DOF adjoint record co-optimization (4 FSG events)
  3. Orthogonal suspension architecture physics benchmark & mass break-even
  4. Stochastic Tube-MPC asymmetric stress-testing (Split-mu & 3WD Limp-Home)
  5. MirenaSim (Godot 4 Full 14-DOF) Sim-to-Sim parity + 5x 300-DPI pure-ASCII plots
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DEEP_VAL_DIR = REPO_ROOT / "outputs" / "differentiable_deep_validation"


def _load_json(filename: str) -> dict:
    p = DEEP_VAL_DIR / filename
    assert p.exists(), f"Missing report: {p}"
    raw = p.read_text(encoding="utf-8")
    assert all(ord(ch) < 128 for ch in raw), f"Non-ASCII character found in {p}"
    return json.loads(raw)


def test_ter26_adjoint_calibration_and_c2_audit() -> None:
    d = _load_json("ter26_adjoint_calibration_report.json")
    audit = d["c2_smoothing_and_gradient_audit"]
    assert int(audit["model_14dof_smoothing_sensitivity"]["smoothing_error_lt_1e4"]) == 1
    assert float(audit["model_14dof_smoothing_sensitivity"]["calibrated_vs_0p1x_tight_ay_rel_err"]) < 1e-4
    assert int(audit["adjoint_vs_central_fd_19params"]["all_passed_lt_1e6"]) == 1
    assert float(audit["adjoint_vs_central_fd_19params"]["max_rel_err"]) < 1e-6

    s_all = d["before_vs_after_summary"]["all_5_runs"]
    assert float(s_all["wz_rads"]["after_r"]) >= 0.985
    assert float(s_all["wz_rads"]["after_rmse"]) < float(s_all["wz_rads"]["before_rmse"])
    assert float(s_all["ay_ms2"]["after_rmse"]) < float(s_all["ay_ms2"]["before_rmse"])
    assert float(s_all["ax_ms2"]["after_rmse"]) < float(s_all["ax_ms2"]["before_rmse"])
    assert float(s_all["vx_ms"]["after_rmse"]) < float(s_all["vx_ms"]["before_rmse"])

    s_val = d["before_vs_after_summary"]["held_out_val_2_runs"]
    assert float(s_val["wz_rads"]["after_rmse"]) < float(s_val["wz_rads"]["before_rmse"])
    assert float(s_val["ay_ms2"]["after_rmse"]) < float(s_val["ay_ms2"]["before_rmse"])


def test_ter27_warmstart_adjoint_record_optimization() -> None:
    d = _load_json("ter27_record_adjoint_opt_report.json")
    assert d["all_constraints_certified"] is True

    fd = d["adjoint_fd_verification"]
    for ev_k in ("skidpad", "acceleration", "autocross", "endurance"):
        assert ev_k in d["events"]
        ev = d["events"][ev_k]
        assert ev["constraints_passed"] is True
        assert float(ev["optimized_14dof"]["event_time_s"]) < float(ev["baseline_14dof"]["event_time_s"])
        if ev_k in fd:
            err_val = fd[ev_k] if isinstance(fd[ev_k], (int, float)) else fd[ev_k].get("max_rel_err", 0.0)
            assert float(err_val) < 1e-6

    # Verify all 4 exported MirenaSim setup JSONs exist and are pure ASCII
    setup_dir = DEEP_VAL_DIR / "adjoint_optimized_setups"
    for fn in ("adjoint_skidpad.json", "adjoint_accel.json", "adjoint_autocross.json", "adjoint_endurance.json"):
        p = setup_dir / fn
        assert p.exists(), f"Missing exported adjoint setup: {p}"
        txt = p.read_text(encoding="utf-8")
        assert all(ord(ch) < 128 for ch in txt)
        sd = json.loads(txt)
        assert sd["vehicle_id"] == "ter27_4wd"
        assert sd["adjoint_metrics"]["hv_violations"] == 0
        assert sd["adjoint_metrics"]["bspd_violations"] == 0


def test_suspension_architectures_benchmark() -> None:
    d = _load_json("suspension_architecture_benchmark_report.json")
    inv = d["safety_invariants"]
    assert inv["all_hv_dc_within_minus25_plus80_kw"] is True
    assert inv["all_bspd_zero_violations"] is True
    assert inv["all_skidpad_zero_cones"] is True
    assert inv["all_skidpad_min_clear_ge_0p150m"] is True
    assert inv["all_yaw_stable_dmz_dbeta_lt_0"] is True

    ax_rows = {r["architecture"]: r for r in d["comparative_tables"]["autocross"]}
    assert set(ax_rows.keys()) >= {
        "analytic_1d_lut",
        "3d_double_wishbone_lut",
        "decoupled_heave_roll",
        "standard_pushrod",
        "decoupled_heave_roll_tuned",
    }
    assert float(ax_rows["decoupled_heave_roll_tuned"]["event_time_s"]) < float(
        ax_rows["standard_pushrod"]["event_time_s"]
    )
    be_tuned = d["mass_penalty_breakeven"]["decoupled_heave_roll_tuned"]["breakeven_mass_kg"]
    assert float(be_tuned["vs_standard_pushrod_autocross"]) > 5.0


def test_tube_mpc_asymmetric_stress_invariants() -> None:
    d = _load_json("tube_mpc_asymmetric_stress_report.json")
    gates = d["acceptance_gates"]
    assert gates["overall_pass"] is True
    assert gates["all_scenarios_refined_zero_cones_and_min_clear_ge_150mm"] is True
    assert gates["all_scenarios_refined_zero_hv_and_bspd_violations"] is True
    assert gates["all_scenarios_refined_strict_dmz_dbeta_lt_0"] is True

    gp = d["grip_profiles_preserved"]
    assert abs(float(gp["FSG_RUBBERED"]) - 0.57) < 1e-9
    assert abs(float(gp["COLD_TEST_OCT"]) - 0.47) < 1e-9

    for sc_k in (
        "scenario_a_split_mu_autocross_corner",
        "scenario_b1_3wd_limp_home_braking_entry",
        "scenario_b2_3wd_limp_home_slalom_transition",
    ):
        ref_sum = d["scenarios"][sc_k]["controllers"]["refined_asymmetric_tube_mpc"]["summary"]
        assert int(ref_sum["cones_hit"]) == 0
        assert float(ref_sum["min_cone_clearance_m"]) >= 0.150
        assert int(ref_sum["hv_violations"]) == 0
        assert int(ref_sum["bspd_violations"]) == 0
        assert float(ref_sum["max_dMz_dbeta"]) < 0.0


def test_mirenasim_godot4_sim2sim_parity_and_plots() -> None:
    d = _load_json("mirenasim_sim2sim_parity_report.json")
    assert float(d["ter26_sim2sim_parity"]["casadi_vs_godot4"]["ay_pearson_r"]) >= 0.95

    t27 = d["ter27_baseline_vs_adjoint_godot4"]
    for ev_k in ("skidpad", "accel", "autocross", "endurance"):
        adj = t27[ev_k]["godot4_adjoint"]
        assert bool(adj["completed"]) is True
        assert int(adj["hv_violations"]) == 0
        assert float(adj["min_cone_clear_m"]) >= 0.150
        assert float(adj["min_rh_f_mm"]) > 10.0

    # Acceleration Sim-to-Sim CasADi vs Godot 4 within 2.5%
    assert abs(float(t27["accel"]["sim2sim_time_delta_pct"])) < 2.5

    # Extreme stress scenarios in Godot 4 with SENSOR_TRACK_REALISM active
    st = d["extreme_stress_godot4_sensor_realism"]
    for sc_k in ("scenario_a_split_mu", "scenario_b1_limp_braking", "scenario_b2_limp_slalom"):
        tube = st[sc_k]["godot4_refined_tube_mpc"]
        assert int(tube["cones_hit"]) == 0
        assert float(tube["min_cone_clear_m"]) >= 0.150
        assert int(tube["hv_violations"]) == 0

    # Verify all 5 300-DPI PNG plots exist and are non-trivial (> 30 KB)
    expected_figs = [
        "fig01_ter26_adjoint_calibration_before_after.png",
        "fig02_ter27_record_lap_ceiling_and_attribution.png",
        "fig03_suspension_architectures_benchmark.png",
        "fig04_tube_mpc_split_mu_and_3wd_limp_home.png",
        "fig05_mirenasim_godot4_sim2sim_parity.png",
    ]
    for fn in expected_figs:
        fp = DEEP_VAL_DIR / fn
        assert fp.exists(), f"Missing plot: {fp}"
        assert fp.stat().st_size > 30_000, f"Plot too small: {fp}"
        with Image.open(fp) as im:
            dpi = im.info.get("dpi", (0, 0))
            assert dpi[0] >= 299 and dpi[1] >= 299, f"Plot {fn} DPI={dpi} < 300"

    # Verify executive SUMMARY_REPORT.json
    summary = _load_json("SUMMARY_REPORT.json")
    assert summary["all_gates_passed"] is True
