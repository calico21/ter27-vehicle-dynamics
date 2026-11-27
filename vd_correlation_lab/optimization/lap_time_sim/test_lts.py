#!/usr/bin/env python3
"""Automated verification tests for MirenaSim Hybrid LTS (Phase 5)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = LAB_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.optimization.lap_time_sim import (
    Track,
    extract_or_load_ggv,
    optimize_mintime_racing_line,
    optimize_racing_line,
    run_transient_lap,
    search_fastest_valid_lap,
    solve_qss,
)
from vd_correlation_lab.optimization.lap_time_sim.tracks.generate_tracks import ensure_all_reference_tracks


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_tracks_and_racing_line() -> dict:
    tracks = ensure_all_reference_tracks()
    _assert("fsg_autocross_endurance" in tracks and "mirena_track10" in tracks, "Both reference tracks missing")

    rls = {}
    for name, trk in tracks.items():
        val = trk.check_validity()
        _assert(bool(val["valid"]), f"Track {name} failed validity: {val}")
        _assert(float(val["min_width_m"]) >= 2.999, f"Track {name} narrower than 3.0 m FS minimum")

        rl = optimize_racing_line(trk, car_track_width_m=1.230, safety_margin_m=0.35)
        rval = rl.racing_track.check_validity()
        _assert(rl.within_bounds, f"Racing line {name} exceeded strict lateral bounds")
        _assert(bool(rval["valid"]), f"Racing line track {name} invalid: {rval}")
        _assert(
            rl.stats["int_k2_reduction_pct"] > 5.0,
            f"Racing line {name} failed to reduce int(kappa^2 ds): {rl.stats['int_k2_reduction_pct']}",
        )
        rls[name] = rl
        print(
            f"[PASS] Track & RacingLine [{name}]: L={trk.length:.1f}m -> {rl.racing_track.length:.1f}m, "
            f"-{rl.stats['int_k2_reduction_pct']:.1f}% int(k^2), R_min={rl.stats['r_min_racing_m']:.2f}m"
        )
    return rls


def test_ggv_and_qss(rls: dict) -> tuple:
    ggv = extract_or_load_ggv(label="tv4_auto", ecu_driving_mode=2, tv_4wd_mode=0, use_3d_kinematics=True)
    # Monotonicity and physical power/friction limits
    _assert(bool(np.all(np.diff(ggv.ay_max) >= -1e-6)), f"ay_max not monotonic: {ggv.ay_max}")
    _assert(float(ggv.ay_max[-1]) > float(ggv.ay_max[0]), "ay_max should grow with aero downforce")
    _assert(bool(np.all(np.diff(ggv.ax_max, axis=1) <= 1e-6)), "ax_max must decrease with lateral g")
    _assert(bool(np.all(np.diff(ggv.ax_min, axis=1) >= -1e-6)), "ax_min must shrink in magnitude with lateral g")
    _assert(float(np.max(ggv.p_acc_kw)) <= 80.5, f"GGV peak power exceeded 80.5 kW: {np.max(ggv.p_acc_kw)}")
    _assert(float(np.min(ggv.p_brk_kw)) >= -25.5, f"GGV regen power exceeded -25.5 kW: {np.min(ggv.p_brk_kw)}")

    rl_fsg = rls["fsg_autocross_endurance"]
    qss1 = solve_qss(rl_fsg.racing_track, ggv)
    qss2 = solve_qss(rl_fsg.racing_track, ggv)
    _assert(qss1.solve_time_ms < 50.0, f"QSS solve_time_ms={qss1.solve_time_ms:.2f} ms >= 50 ms")
    _assert(abs(qss1.lap_time_s - qss2.lap_time_s) < 1e-12, "QSS not deterministic")
    _assert(bool(np.allclose(qss1.v_qss, qss2.v_qss)), "QSS velocity trace not deterministic")
    _assert(qss1.v_max_kmh > 90.0 and qss1.v_min_kmh < 60.0, f"Unexpected QSS speed bounds: {qss1.v_min_kmh:.1f}..{qss1.v_max_kmh:.1f}")
    print(
        f"[PASS] GGV & QSS: ay_max=[{ggv.ay_max[0]:.2f}..{ggv.ay_max[-1]:.2f}] m/s^2, "
        f"lap={qss1.lap_time_s:.3f}s in {qss1.solve_time_ms:.2f}ms (< 50 ms), vmax={qss1.v_max_kmh:.1f} km/h"
    )
    return ggv, qss1


def test_transient_parity_and_multilap(rls: dict, ggv, qss_fsg) -> None:
    rl_fsg = rls["fsg_autocross_endurance"]
    res, _ = search_fastest_valid_lap(
        label="test_tv4_auto",
        racing_line=rl_fsg,
        qss=qss_fsg,
        ecu_driving_mode=2,
        tv_4wd_mode=0,
        n_laps=1,
        use_3d_kinematics=True,
    )
    s = res.summary
    _assert(res.valid, f"Stage-2 lap invalid: {s}")
    _assert(int(s["off_track_ticks"]) == 0, f"Off-track ticks > 0: {s['off_track_ticks']}")
    _assert(int(s["cones_hit"]) == 0, f"Cones hit > 0: {s['cones_hit']}")
    _assert(float(s["max_abs_beta_deg"]) < 15.0, f"Excessive beta: {s['max_abs_beta_deg']}")
    _assert(res.lap_time_s >= 0.98 * qss_fsg.lap_time_s, f"Transient {res.lap_time_s:.3f}s unrealistically faster than QSS {qss_fsg.lap_time_s:.3f}s")
    _assert(float(s["p_dc_max_kw"]) <= 80.5, f"Peak DC power {s['p_dc_max_kw']:.2f} > 80.5 kW")
    _assert(float(s["p_chg_min_kw"]) >= -25.5, f"Peak regen power {s['p_chg_min_kw']:.2f} < -25.5 kW")
    print(
        f"[PASS] QSS vs 14-DOF Parity: QSS={qss_fsg.lap_time_s:.3f}s vs 14DOF={res.lap_time_s:.3f}s "
        f"(ratio={res.lap_time_s/qss_fsg.lap_time_s:.4f}), P_dc=[{s['p_chg_min_kw']:.1f}, {s['p_dc_max_kw']:.1f}] kW"
    )

    # Multi-lap continuous stability test (2 continuous laps without resets)
    alpha_opt = float(s["alpha_drv"])
    ml = run_transient_lap(
        label="test_multilap_eff",
        racing_line=rl_fsg,
        qss=qss_fsg,
        ecu_driving_mode=2,
        tv_4wd_mode=2,
        alpha_drv=min(alpha_opt, 0.99),
        n_laps=2,
        use_3d_kinematics=True,
    )
    ms = ml.summary
    _assert(ml.valid and not bool(ms["nan_detected"]), f"Multi-lap instability/invalidity: {ms}")
    _assert(len(ms["per_lap"]) == 2, f"Expected 2 completed laps, got {len(ms['per_lap'])}")
    for lp in ms["per_lap"]:
        _assert(math.isfinite(float(lp["mz_int_nm"])), "NaN/Inf in _mz_int")
        _assert(math.isfinite(float(lp["rls_theta_0"])), "NaN/Inf in rls_theta")
        _assert(0.60 <= float(lp["mu_hat_min"]) <= float(lp["mu_hat_max"]) <= 1.56, f"mu_hat out of physical bounds: {lp}")
        _assert(abs(float(lp["ekf_vx_err_ms"])) < 1.0, f"EKF drift detected: {lp['ekf_vx_err_ms']}")
    t_l1 = float(ms["per_lap"][0]["lap_time_s"])
    t_l2 = float(ms["per_lap"][1]["lap_time_s"])
    _assert(abs(t_l1 - t_l2) < 1.0, f"Large multi-lap lap-time drift: {t_l1:.3f}s vs {t_l2:.3f}s")
    print(f"[PASS] Multi-Lap Stability (Efficiency 2 Laps): Lap1={t_l1:.3f}s, Lap2={t_l2:.3f}s, EKF/RLS/mu_hat healthy")


def test_mintime_ocp(ggv) -> None:
    fsg_p = Path(__file__).resolve().parent / "tracks" / "fsg_2023.json"
    _assert(fsg_p.exists(), f"Missing track: {fsg_p}")
    trk = Track.load_json(fsg_p)
    rl_old = optimize_racing_line(trk, car_track_width_m=1.230, safety_margin_m=0.35)
    rl_mt = optimize_mintime_racing_line(trk, ggv=ggv)
    rval = rl_mt.racing_track.check_validity()
    _assert(rl_mt.within_bounds, "MT-OCP exceeded strict lateral bounds |n| <= n_max")
    _assert(bool(rval["valid"]), f"MT-OCP track invalid: {rval}")
    _assert(float(rl_mt.stats["corridor_util_pct"]) >= 85.0, f"MT-OCP corridor util < 85%: {rl_mt.stats['corridor_util_pct']}")
    q_c = solve_qss(trk, ggv)
    q_old = solve_qss(rl_old.racing_track, ggv)
    q_mt = solve_qss(rl_mt.racing_track, ggv)
    _assert(q_mt.lap_time_s < q_old.lap_time_s - 0.5, f"MT-OCP QSS ({q_mt.lap_time_s:.3f}s) did not beat Min-Curv ({q_old.lap_time_s:.3f}s)")
    _assert(q_mt.lap_time_s < q_c.lap_time_s - 2.0, f"MT-OCP QSS ({q_mt.lap_time_s:.3f}s) did not beat centerline ({q_c.lap_time_s:.3f}s)")
    print(
        f"[PASS] MT-OCP [fsg_2023]: QSS={q_c.lap_time_s:.3f}s (center) -> {q_old.lap_time_s:.3f}s (old) -> {q_mt.lap_time_s:.3f}s (MT-OCP), "
        f"corridor={rl_mt.stats['corridor_util_pct']:.1f}%, R_min={rl_mt.stats['r_min_racing_m']:.2f}m"
    )


def main() -> None:
    rls = test_tracks_and_racing_line()
    ggv, qss_fsg = test_ggv_and_qss(rls)
    test_mintime_ocp(ggv)
    test_transient_parity_and_multilap(rls, ggv, qss_fsg)
    print("\nALL LTS TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
