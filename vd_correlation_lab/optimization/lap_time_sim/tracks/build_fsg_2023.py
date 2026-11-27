#!/usr/bin/env python3
"""
Build the real FSG 2023 (Hockenheim) track from the 160 calibrated waypoints:
  1. vd_correlation_lab/lts/tracks/fsg_2023.json (LTS parametric track, ds = 0.50 m, width = 3.60 m)
  2. src/mirena_sim/TrackFiles/fsg_2023.json (playable cone track in MirenaSim OpenTrack)
  3. src/mirena_planning/test/tracks/fsg_2023.json (matching track10.json folder & schema)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

import sys
TRACKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRACKS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.optimization.lap_time_sim.track import Track, build_track_from_waypoints  # noqa: E402

WAYPOINTS_FSG_2023 = [
    (0.00, 0.00), (0.67, 7.41), (4.13, 14.04), (8.71, 19.99), (13.67, 25.61), (19.70, 30.01), (26.75, 32.55), (30.84, 38.53),
    (32.84, 45.74), (37.42, 51.63), (43.24, 56.36), (49.29, 60.79), (55.27, 65.33), (61.37, 69.69), (67.82, 73.51), (75.10, 74.63),
    (82.02, 71.77), (89.31, 70.26), (96.77, 70.98), (104.27, 71.19), (111.77, 71.43), (118.88, 73.59), (123.63, 79.30), (127.19, 85.90),
    (131.12, 92.28), (137.22, 96.46), (144.47, 95.51), (149.92, 90.47), (153.67, 83.98), (159.08, 78.84), (165.63, 75.20), (172.41, 71.98),
    (179.69, 70.61), (185.99, 74.37), (188.87, 81.22), (190.37, 88.57), (192.28, 95.83), (194.66, 102.94), (198.77, 109.09), (205.86, 110.33),
    (211.80, 105.85), (216.38, 99.93), (219.15, 92.98), (219.54, 85.59), (218.04, 78.23), (216.15, 70.97), (213.79, 63.85), (210.89, 56.94),
    (207.14, 50.47), (200.92, 46.51), (193.60, 44.87), (187.28, 41.00), (184.06, 34.32), (182.96, 26.90), (181.61, 19.54), (178.31, 12.82),
    (174.18, 6.55), (169.82, 0.45), (165.33, -5.56), (160.00, -10.83), (154.10, -15.47), (148.92, -20.87), (144.56, -26.97), (140.36, -33.19),
    (135.46, -38.86), (129.50, -43.40), (123.00, -47.13), (115.74, -48.73), (109.77, -44.87), (112.26, -38.23), (118.06, -33.48), (123.90, -28.78),
    (129.44, -23.71), (134.53, -18.21), (138.77, -12.03), (140.95, -4.89), (141.84, 2.56), (143.94, 9.65), (149.38, 14.76), (155.14, 19.52),
    (157.55, 26.47), (157.01, 33.95), (158.98, 41.04), (163.88, 46.72), (165.18, 53.86), (161.26, 60.15), (155.25, 64.62), (148.81, 68.46),
    (141.64, 70.33), (134.87, 67.36), (130.16, 61.58), (127.44, 54.60), (126.98, 47.15), (127.05, 39.67), (124.98, 32.48), (121.68, 25.74),
    (117.91, 19.26), (113.73, 13.03), (109.41, 6.89), (104.48, 1.25), (98.43, -3.12), (91.10, -3.55), (85.38, 1.07), (83.37, 8.20),
    (83.66, 15.69), (86.12, 22.74), (91.18, 28.19), (97.93, 31.40), (104.79, 34.39), (109.71, 39.91), (108.88, 47.04), (103.58, 52.32),
    (98.15, 57.50), (91.91, 61.44), (84.83, 59.55), (79.86, 54.04), (76.43, 47.37), (72.08, 41.26), (68.97, 34.47), (66.58, 27.38),
    (62.37, 21.18), (58.80, 14.60), (56.81, 7.40), (58.17, 0.12), (62.39, -6.06), (67.95, -11.08), (74.10, -15.37), (78.55, -21.23),
    (76.70, -28.21), (70.05, -31.19), (62.69, -29.91), (55.52, -27.67), (48.69, -24.63), (43.51, -19.27), (40.20, -12.55), (35.01, -7.32),
    (27.81, -7.89), (22.66, -13.19), (21.97, -20.53), (24.63, -27.51), (28.93, -33.65), (34.14, -39.02), (40.59, -42.82), (47.42, -45.94),
    (54.06, -49.40), (57.93, -55.57), (55.84, -62.61), (50.49, -67.81), (43.45, -70.06), (36.18, -68.40), (30.04, -64.16), (25.17, -58.46),
    (20.63, -52.49), (16.21, -46.42), (11.90, -40.28), (7.42, -34.26), (2.92, -28.26), (-1.02, -21.88), (-2.51, -14.64), (-0.48, -7.43),
]

LTS_OUT = TRACKS_DIR / "fsg_2023.json"
PLANNING_OUT = REPO_ROOT / "src" / "mirena_planning" / "test" / "tracks" / "fsg_2023.json"


def _smooth_periodic_if_needed(pts: np.ndarray, min_radius_req: float = 4.2) -> np.ndarray:
    """Apply minimal periodic Gaussian smoothing on resampled spline only until R_min >= min_radius_req."""
    t_raw = build_track_from_waypoints("fsg_raw", pts, 1.80, 1.80, True, 0.50)
    st0 = t_raw.check_validity(min_clearance_m=7.5)
    if float(st0["min_radius_m"]) >= min_radius_req:
        return pts
    for sigma_m in (1.0, 2.0, 2.6, 3.0, 3.4):
        sig_idx = sigma_m / t_raw.ds
        xs = gaussian_filter1d(t_raw.x, sigma=sig_idx, mode="wrap")
        ys = gaussian_filter1d(t_raw.y, sigma=sig_idx, mode="wrap")
        xs -= xs[0]
        ys -= ys[0]
        cand_pts = np.column_stack([xs, ys])[::4]
        t_cand = build_track_from_waypoints("fsg_chk", cand_pts, 1.80, 1.80, True, 0.50)
        st = t_cand.check_validity(min_clearance_m=7.5)
        if float(st["min_radius_m"]) >= min_radius_req:
            return cand_pts
    return cand_pts


def build_fsg_2023_track(ds_target: float = 0.50) -> Track:
    pts = np.asarray(WAYPOINTS_FSG_2023, dtype=np.float64)
    pts_sm = _smooth_periodic_if_needed(pts, min_radius_req=4.2)
    trk = build_track_from_waypoints(
        name="fsg_2023",
        waypoints_xy=pts_sm,
        w_left=1.80,
        w_right=1.80,
        is_closed=True,
        ds_target=ds_target,
    )
    return trk


def build_cone_track_dict(trk: Track, gate_spacing_m: float = 4.5) -> dict:
    """Export playable MirenaSim cone track JSON (ISO/ROS XY coordinates)."""
    n_gates = max(20, int(round(trk.length / gate_spacing_m)))
    s_gates = np.linspace(0.0, trk.length, n_gates, endpoint=False)

    # Periodic CubicSpline over closed track for exact smooth sampling
    s_wrap = np.append(trk.s, trk.length)
    x_wrap = np.append(trk.x, trk.x[0])
    y_wrap = np.append(trk.y, trk.y[0])
    cs_x = CubicSpline(s_wrap, x_wrap, bc_type="periodic")
    cs_y = CubicSpline(s_wrap, y_wrap, bc_type="periodic")

    cones: list[dict] = []

    # Big orange double start/finish gate at s = 0.0 and s = 0.6 m
    for s_o in (0.0, 0.60):
        xo = float(cs_x(s_o))
        yo = float(cs_y(s_o))
        dxo = float(cs_x(s_o, 1))
        dyo = float(cs_y(s_o, 1))
        nrm = max(np.hypot(dxo, dyo), 1e-9)
        nx, ny = -dyo / nrm, dxo / nrm
        # Left & right big orange cones
        cones.append({"type": "cone_big_orange", "x": round(xo + 1.80 * nx, 3), "y": round(yo + 1.80 * ny, 3)})
        cones.append({"type": "cone_big_orange", "x": round(xo - 1.80 * nx, 3), "y": round(yo - 1.80 * ny, 3)})

    # Track cones every ~4.5 m (skip gate near s=0 where big orange cones are placed)
    path_points: list[dict] = []
    for k, sg in enumerate(s_gates):
        xg = float(cs_x(sg))
        yg = float(cs_y(sg))
        dxg = float(cs_x(sg, 1))
        dyg = float(cs_y(sg, 1))
        nrm = max(np.hypot(dxg, dyg), 1e-9)
        nx, ny = -dyg / nrm, dxg / nrm
        path_points.append({"x": round(xg, 3), "y": round(yg, 3)})
        if sg < 2.0 or sg > trk.length - 2.0:
            continue
        cones.append({"type": "cone_blue", "x": round(xg + 1.80 * nx, 3), "y": round(yg + 1.80 * ny, 3)})
        cones.append({"type": "cone_yellow", "x": round(xg - 1.80 * nx, 3), "y": round(yg - 1.80 * ny, 3)})

    return {
        "metadata": {
            "track_name": "FSG 2023 (Hockenheim)",
            "description": "Real Formula Student Germany 2023 Autocross/Endurance layout (3.60 m width)",
            "date": date.today().isoformat(),
        },
        "setup": {
            "car_start_pose": {
                "x": 0.0,
                "y": 0.0,
                "psi": round(float(trk.psi[0]), 6),
            }
        },
        "cones": cones,
        "path": {
            "closed": True,
            "points": path_points,
        },
    }


def main() -> None:
    trk = build_fsg_2023_track(ds_target=0.50)
    st = trk.check_validity(min_clearance_m=7.5)

    assert st["closure_pos_err_m"] < 0.01, f"closure_pos_err_m too high: {st['closure_pos_err_m']}"
    assert st["closure_psi_err_rad"] < 1e-4, f"closure_psi_err_rad too high: {st['closure_psi_err_rad']}"
    assert st["min_nonadj_clearance_m"] >= 7.5, f"clearance too small: {st['min_nonadj_clearance_m']}"
    assert st["min_radius_m"] >= 4.2, f"min_radius_m < 4.2 m: {st['min_radius_m']}"

    trk.save_json(LTS_OUT)
    cone_dict = build_cone_track_dict(trk, gate_spacing_m=4.5)
    PLANNING_OUT.parent.mkdir(parents=True, exist_ok=True)
    PLANNING_OUT.write_text(json.dumps(cone_dict, indent=2), encoding="utf-8")

    print("FSG 2023 LTS saved ->", LTS_OUT)
    print("FSG 2023 Planning cones saved ->", PLANNING_OUT)
    print("Validity metrics:", json.dumps(st, indent=2))
    print(f"Cones count: {len(cone_dict['cones'])}, Path points: {len(cone_dict['path']['points'])}, Start psi: {cone_dict['setup']['car_start_pose']['psi']} rad")


if __name__ == "__main__":
    main()
