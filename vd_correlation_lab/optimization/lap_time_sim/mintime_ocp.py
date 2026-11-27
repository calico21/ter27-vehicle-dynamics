"""
Minimum-Time Optimal Control Racing Line Optimizer (MT-OCP) for MirenaSim (TeR27 4WD).

Formulation:
  - Curvilinear Frenet coordinates along periodic centerline r_r(s) = r_c(s) + n(s) * m(s)
  - Strict physical lateral corridor bounds:
        -(w_R(s) - 0.5*w_car - margin) <= n(s) <= (w_L(s) - 0.5*w_car - margin)
    Default TeR27 physical envelope: w_car = 1.42 m (half-width 0.71 m), safety_margin = 0.14 m
    -> n_margin = 0.85 m (gives n_max = +/-0.95 m on 3.60 m FSG 2023, +/-0.65 m on 3.00 m Track 10).
  - Stage A: Full-corridor exact Minimum-Curvature SQP initialization (zero centering penalty,
    tiny d(kappa)/ds regularization only).
  - Stage B: Direct Minimum-Time optimization minimizing exact coupled QSS-GGV lap time
        J(n) = T_lap^GGV(r_c + n * m) + eps_dk * integral (d(kappa)/ds)^2 ds
    over periodic C^2 B-spline control nodes using L-BFGS-B with graph-colored exact windowed
    sensitivities (exploiting local braking/acceleration horizon), naturally producing late apexes
    before long straights and outer-inner-outer sweeps across the full usable corridor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, diags, lil_matrix

from .ggv_extractor import GGVMap
from .qss_solver import QSSResult, solve_qss
from .racing_line import RacingLineResult
from .track import Track, build_track_from_waypoints


def _build_diff_matrices(n: int, ds: float, is_closed: bool) -> Tuple[csc_matrix, csc_matrix, csc_matrix]:
    """Build sparse 1st, 2nd, and 3rd derivative matrices along arc-length grid."""
    inv_ds = 1.0 / max(ds, 1e-6)
    inv_ds2 = inv_ds * inv_ds
    D1 = lil_matrix((n, n), dtype=np.float64)
    D2 = lil_matrix((n, n), dtype=np.float64)
    for i in range(n):
        if is_closed:
            im1 = (i - 1) % n
            ip1 = (i + 1) % n
            D1[i, im1] = -0.5 * inv_ds
            D1[i, ip1] = 0.5 * inv_ds
            D2[i, im1] = inv_ds2
            D2[i, i] = -2.0 * inv_ds2
            D2[i, ip1] = inv_ds2
        else:
            if 0 < i < n - 1:
                D1[i, i - 1] = -0.5 * inv_ds
                D1[i, i + 1] = 0.5 * inv_ds
                D2[i, i - 1] = inv_ds2
                D2[i, i] = -2.0 * inv_ds2
                D2[i, i + 1] = inv_ds2
            elif i == 0:
                D1[i, 0] = -inv_ds
                D1[i, 1] = inv_ds
                D2[i, 0] = inv_ds2
                D2[i, 1] = -2.0 * inv_ds2
                D2[i, 2] = inv_ds2
            else:
                D1[i, n - 2] = -inv_ds
                D1[i, n - 1] = inv_ds
                D2[i, n - 3] = inv_ds2
                D2[i, n - 2] = -2.0 * inv_ds2
                D2[i, n - 1] = inv_ds2
    D1c = D1.tocsc()
    D2c = D2.tocsc()
    D3c = (D1c @ D2c).tocsc()
    return D1c, D2c, D3c


def _eval_curve_fast(
    track: Track,
    n_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast exact displaced curve geometry r_r = r_c + n * m:
    Returns (xr, yr, psi_r, kappa_r, ds_seg).
    """
    nx = -np.sin(track.psi)
    ny = np.cos(track.psi)
    xr = track.x + n_vec * nx
    yr = track.y + n_vec * ny
    if track.is_closed:
        xw = np.append(xr, xr[0])
        yw = np.append(yr, yr[0])
        seg = np.hypot(np.diff(xw), np.diff(yw))
        seg = np.maximum(seg, 1e-4)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        csx = CubicSpline(t, xw, bc_type="periodic")
        csy = CubicSpline(t, yw, bc_type="periodic")
        te = t[:-1]
    else:
        seg = np.hypot(np.diff(xr), np.diff(yr))
        seg = np.maximum(seg, 1e-4)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        csx = CubicSpline(t, xr, bc_type="not-a-knot")
        csy = CubicSpline(t, yr, bc_type="not-a-knot")
        te = t
    dx = csx(te, 1)
    dy = csy(te, 1)
    ddx = csx(te, 2)
    ddy = csy(te, 2)
    sp2 = np.maximum(dx * dx + dy * dy, 1e-12)
    psi_r = np.arctan2(dy, dx)
    kr = (dx * ddy - dy * ddx) / (sp2 ** 1.5)
    return xr, yr, psi_r, kr, seg


def _build_periodic_bspline_basis(n_grid: int, n_ctrl: int, is_closed: bool) -> np.ndarray:
    """
    Build uniform cubic B-spline interpolation matrix B of shape (n_grid, n_ctrl).
    Partition of unity: row sums == 1.0 and B >= 0, so if lb_c <= c <= ub_c then
    min(lb_c) <= (B @ c)_i <= max(ub_c) everywhere automatically.
    """
    B = np.zeros((n_grid, n_ctrl), dtype=np.float64)
    if is_closed:
        u_grid = np.linspace(0.0, float(n_ctrl), n_grid, endpoint=False)
        for i, u in enumerate(u_grid):
            k = int(np.floor(u))
            t = u - k
            t2 = t * t
            t3 = t2 * t
            w0 = ((1.0 - t) ** 3) / 6.0
            w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
            w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
            w3 = t3 / 6.0
            B[i, (k - 1) % n_ctrl] += w0
            B[i, k % n_ctrl] += w1
            B[i, (k + 1) % n_ctrl] += w2
            B[i, (k + 2) % n_ctrl] += w3
    else:
        u_grid = np.linspace(0.0, float(n_ctrl - 1), n_grid, endpoint=True)
        for i, u in enumerate(u_grid):
            k = min(int(np.floor(u)), n_ctrl - 2)
            t = u - k
            t2 = t * t
            t3 = t2 * t
            w0 = ((1.0 - t) ** 3) / 6.0
            w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
            w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
            w3 = t3 / 6.0
            B[i, max(k - 1, 0)] += w0
            B[i, k] += w1
            B[i, min(k + 1, n_ctrl - 1)] += w2
            B[i, min(k + 2, n_ctrl - 1)] += w3
    return B


def _solve_full_corridor_min_curv(
    track: Track,
    lb: np.ndarray,
    ub: np.ndarray,
    dk_reg: float = 2.0e-4,
    n_sqp_iters: int = 12,
) -> np.ndarray:
    """
    Stage A: Exact full-corridor minimum-curvature SQP (Heilmeier/TUMFTM style).
    Zero centering penalty on |n|; only minimal d(kappa)/ds regularization to prevent chattering.
    """
    n = track.n_points
    D1, D2, _ = _build_diff_matrices(n, track.ds, track.is_closed)
    k_c = track.kappa
    ds0 = track.ds

    n_cur = np.zeros(n, dtype=np.float64)
    _, _, _, k_cur, seg_cur = _eval_curve_fast(track, n_cur)

    def _true_obj(k_arr: np.ndarray, seg_arr: np.ndarray) -> float:
        w = seg_arr[:n]
        dk = D1 @ k_arr
        return float(np.sum(w * k_arr * k_arr) + dk_reg * np.sum(w * dk * dk))

    best_n = n_cur.copy()
    best_obj = _true_obj(k_cur, seg_cur)

    for _it in range(n_sqp_iters):
        denom = np.maximum(1.0 - k_c * n_cur, 0.20)
        inv_d2 = diags(1.0 / (denom * denom), 0, format="csc")
        K2_diag = diags(k_c * k_c, 0, format="csc")
        J_k = inv_d2 @ (D2 + K2_diag)
        J_kT = J_k.T.tocsc()
        J_dk = (D1 @ J_k).tocsc()
        J_dkT = J_dk.T.tocsc()
        dk_cur = D1 @ k_cur
        w_s = seg_cur[:n]

        bounds = list(zip(lb - n_cur, ub - n_cur))

        def qp_obj_grad(dn: np.ndarray) -> Tuple[float, np.ndarray]:
            k_pred = k_cur + J_k @ dn
            dk_pred = dk_cur + J_dk @ dn
            obj = 0.5 * float(np.sum(w_s * k_pred * k_pred)) + 0.5 * dk_reg * float(np.sum(w_s * dk_pred * dk_pred))
            grad = J_kT @ (w_s * k_pred) + dk_reg * (J_dkT @ (w_s * dk_pred))
            return obj, grad

        res = minimize(
            qp_obj_grad,
            x0=np.zeros(n, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 260, "ftol": 1e-12, "gtol": 1e-8},
        )
        dn = res.x

        step = 1.0
        accepted = False
        for _ in range(8):
            n_try = np.clip(n_cur + step * dn, lb, ub)
            _, _, _, k_try, seg_try = _eval_curve_fast(track, n_try)
            obj_try = _true_obj(k_try, seg_try)
            if obj_try < best_obj - 1e-9:
                n_cur = n_try
                k_cur = k_try
                seg_cur = seg_try
                best_n = n_try.copy()
                best_obj = obj_try
                accepted = True
                break
            step *= 0.5
        if not accepted or np.linalg.norm(step * dn, ord=np.inf) < 2e-4:
            break

    return best_n


def _eval_track_qss_obj(
    track: Track,
    n_vec: np.ndarray,
    ggv: GGVMap,
    dk_reg: float,
    v_top_limit_ms: float = 37.4,
) -> Tuple[float, float, QSSResult, np.ndarray, np.ndarray]:
    """Evaluate exact QSS-GGV lap time + tiny d(kappa)/ds smoothness term."""
    xr, yr, psi_r, kr, seg = _eval_curve_fast(track, n_vec)
    n_pts = track.n_points
    s_r = np.concatenate([[0.0], np.cumsum(seg[:-1])]) if track.is_closed else np.concatenate([[0.0], np.cumsum(seg)])
    len_r = float(np.sum(seg))
    tmp_trk = Track(
        name=track.name,
        s=s_r,
        x=xr,
        y=yr,
        psi=psi_r,
        kappa=kr,
        w_left=track.w_left,
        w_right=track.w_right,
        is_closed=track.is_closed,
        ds=len_r / max(n_pts, 1),
        length=len_r,
    )
    qss = solve_qss(tmp_trk, ggv, v_top_limit_ms=v_top_limit_ms)
    if track.is_closed:
        dk = (np.roll(kr, -1) - np.roll(kr, 1)) / np.maximum(seg + np.roll(seg, 1), 1e-3)
    else:
        dk = np.gradient(kr, s_r)
    reg_term = dk_reg * float(np.sum((dk * dk) * seg[:n_pts]))
    return qss.lap_time_s + reg_term, qss.lap_time_s, qss, kr, seg


def optimize_mintime_racing_line(
    track: Track,
    ggv: Optional[GGVMap] = None,
    car_width_m: float = 1.42,
    safety_margin_m: float = 0.43,
    dk_reg: float = 4.0e-4,
    ctrl_spacing_m: float = 6.0,
    max_opt_iters: int = 22,
    ds_target: float = 0.5,
    v_top_limit_ms: float = 37.4,
) -> RacingLineResult:
    """
    Solve real Minimum-Time Optimal Control Racing Line (MT-OCP):
      Stage A: Full-corridor exact Minimum-Curvature SQP (no centering penalty).
      Stage B (when ggv is provided): Direct Minimum-Time L-BFGS-B optimization over periodic
               C^2 B-spline lateral control nodes using graph-colored windowed exact QSS-GGV
               gradients, generating late apexes before straights and full corridor utilization.
    """
    t0 = time.perf_counter()
    n = track.n_points
    half_car = 0.5 * car_width_m + safety_margin_m  # 0.71 + 0.14 = 0.85 m default
    ub = np.maximum(track.w_left - half_car, 0.0)
    lb = -np.maximum(track.w_right - half_car, 0.0)

    # Prevent centerline-normal self-intersection in extreme curvature corners
    for i in range(n):
        if track.kappa[i] > 1e-4:
            ub[i] = min(ub[i], 0.82 / track.kappa[i])
        elif track.kappa[i] < -1e-4:
            lb[i] = max(lb[i], 0.82 / track.kappa[i])

    # Stage A: Full-corridor minimum-curvature initialization
    t_a0 = time.perf_counter()
    n_init = _solve_full_corridor_min_curv(track, lb, ub, dk_reg=8.0e-5, n_sqp_iters=16)
    best_n = np.clip(n_init, lb, ub)
    stage_a_lap_s = 0.0
    stage_b_lap_s = 0.0
    opt_iters_done = 0
    print(f"  [MT-OCP] Stage A (Min-Curv SQP) done in {time.perf_counter() - t_a0:.2f}s", flush=True)

    if ggv is not None and n >= 20:
        # Stage B: Direct Minimum-Time optimization over periodic cubic B-spline control nodes
        n_ctrl = int(np.clip(round(track.length / max(ctrl_spacing_m, 3.0)), 44, 220))
        B = _build_periodic_bspline_basis(n, n_ctrl, track.is_closed)

        # Fit B-spline control points to Stage A initialization (strictly within bounds)
        c_lb = np.full(n_ctrl, -10.0, dtype=np.float64)
        c_ub = np.full(n_ctrl, 10.0, dtype=np.float64)
        # Tight per-control-node bounds so B @ c strictly satisfies lb <= n <= ub everywhere
        for k in range(n_ctrl):
            supp = B[:, k] > 1e-4
            if np.any(supp):
                c_lb[k] = float(np.max(lb[supp]))
                c_ub[k] = float(np.min(ub[supp]))
            else:
                c_lb[k] = float(np.max(lb))
                c_ub[k] = float(np.min(ub))
        c_lb = np.minimum(c_lb, 0.0)
        c_ub = np.maximum(c_ub, 0.0)

        # Least-squares projection of n_init onto B-spline space, then clip to [c_lb, c_ub]
        BtB = B.T @ B + 1e-6 * np.eye(n_ctrl)
        c0 = np.linalg.solve(BtB, B.T @ n_init)
        c0 = np.clip(c0, c_lb, c_ub)

        n_c0 = np.clip(B @ c0, lb, ub)
        obj_mc, lap_mc, _, _, _ = _eval_track_qss_obj(track, best_n, ggv, dk_reg, v_top_limit_ms)
        obj_c0, lap_c0, _, _, _ = _eval_track_qss_obj(track, n_c0, ggv, dk_reg, v_top_limit_ms)
        stage_a_lap_s = lap_mc
        best_obj = min(obj_mc, obj_c0)
        best_lap = lap_mc if obj_mc <= obj_c0 else lap_c0
        if obj_c0 < obj_mc:
            best_n = n_c0.copy()

        # Precompute graph-coloring groups: spacing ~90 m between simultaneously perturbed nodes
        ctrl_ds = track.length / max(n_ctrl, 1)
        n_colors = int(np.clip(round(90.0 / max(ctrl_ds, 1.0)), 10, min(22, n_ctrl)))
        half_win_pts = int(round(0.5 * (n_colors * ctrl_ds) / max(track.ds, 0.1)))
        ctrl_center_idx = np.argmax(B, axis=0)
        eps_fd = 0.020  # 2.0 cm finite difference step

        eval_cache: Dict[bytes, Tuple[float, np.ndarray]] = {}

        feval_state = {"fevals": 0, "iters": 0}

        def _obj_and_grad(c_vec: np.ndarray) -> Tuple[float, np.ndarray]:
            c_clamped = np.clip(c_vec, c_lb, c_ub)
            key = c_clamped.tobytes()
            if key in eval_cache:
                return eval_cache[key]
            feval_state["fevals"] += 1

            n_base = np.clip(B @ c_clamped, lb, ub)
            obj0, lap0, qss0, kr0, seg0 = _eval_track_qss_obj(track, n_base, ggv, dk_reg, v_top_limit_ms)
            dt0 = qss0.dt_seg
            if track.is_closed:
                dk0 = (np.roll(kr0, -1) - np.roll(kr0, 1)) / np.maximum(seg0 + np.roll(seg0, 1), 1e-3)
            else:
                dk0 = np.gradient(kr0, track.s)
            reg_vec0 = dk_reg * (dk0 * dk0) * seg0[:n]
            tot_vec0 = dt0 + reg_vec0

            grad = np.zeros(n_ctrl, dtype=np.float64)
            for col in range(n_colors):
                idxs = list(range(col, n_ctrl, n_colors))
                if not idxs:
                    continue
                c_pert = c_clamped.copy()
                steps = np.zeros(len(idxs), dtype=np.float64)
                for m_i, k in enumerate(idxs):
                    # Perturb inward if at upper bound, outward otherwise
                    if c_clamped[k] + eps_fd <= c_ub[k]:
                        steps[m_i] = eps_fd
                    elif c_clamped[k] - eps_fd >= c_lb[k]:
                        steps[m_i] = -eps_fd
                    else:
                        steps[m_i] = 0.5 * (c_ub[k] - c_lb[k])
                    c_pert[k] = np.clip(c_clamped[k] + steps[m_i], c_lb[k], c_ub[k])

                n_p = np.clip(B @ c_pert, lb, ub)
                _, _, qss_p, kr_p, seg_p = _eval_track_qss_obj(track, n_p, ggv, dk_reg, v_top_limit_ms)
                dt_p = qss_p.dt_seg
                if track.is_closed:
                    dk_p = (np.roll(kr_p, -1) - np.roll(kr_p, 1)) / np.maximum(seg_p + np.roll(seg_p, 1), 1e-3)
                else:
                    dk_p = np.gradient(kr_p, track.s)
                tot_vec_p = dt_p + dk_reg * (dk_p * dk_p) * seg_p[:n]
                diff_vec = tot_vec_p - tot_vec0

                for m_i, k in enumerate(idxs):
                    st_k = steps[m_i]
                    if abs(st_k) < 1e-6:
                        continue
                    ic = int(ctrl_center_idx[k])
                    if track.is_closed:
                        win_idx = (np.arange(ic - half_win_pts, ic + half_win_pts + 1) % n)
                        d_obj = float(np.sum(diff_vec[win_idx]))
                    else:
                        i_lo = max(0, ic - half_win_pts)
                        i_hi = min(len(diff_vec), ic + half_win_pts + 1)
                        d_obj = float(np.sum(diff_vec[i_lo:i_hi]))
                    grad[k] = d_obj / st_k

            eval_cache[key] = (obj0, grad)
            return obj0, grad

        def _iter_cb(xk: np.ndarray) -> None:
            feval_state["iters"] += 1
            ck = np.clip(xk, c_lb, c_ub)
            obj_k = eval_cache[ck.tobytes()][0] if ck.tobytes() in eval_cache else best_obj
            print(
                f"  [MT-OCP] Stage B iter {feval_state['iters']}/{max_opt_iters} "
                f"(fevals={feval_state['fevals']}, obj={obj_k:.3f}s, {time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

        res_mt = minimize(
            _obj_and_grad,
            x0=c0,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(c_lb, c_ub)),
            callback=_iter_cb,
            options={
                "maxiter": max_opt_iters,
                "maxfun": max_opt_iters * 3 + 10,
                "ftol": 1e-6,
                "gtol": 1e-4,
            },
        )
        opt_iters_done = int(getattr(res_mt, "nit", 0))
        c_opt = np.clip(res_mt.x, c_lb, c_ub)
        n_opt = np.clip(B @ c_opt, lb, ub)
        obj_opt, lap_opt, _, _, _ = _eval_track_qss_obj(track, n_opt, ggv, dk_reg, v_top_limit_ms)
        if obj_opt < best_obj:
            best_n = n_opt.copy()
            best_obj = obj_opt
            best_lap = lap_opt
        stage_b_lap_s = best_lap

    nx = -np.sin(track.psi)
    ny = np.cos(track.psi)
    xr_pts = track.x + best_n * nx
    yr_pts = track.y + best_n * ny

    wl_r = np.maximum(track.w_left - best_n, 0.5 * car_width_m)
    wr_r = np.maximum(track.w_right + best_n, 0.5 * car_width_m)

    racing_track = build_track_from_waypoints(
        name=f"{track.name}_mt_ocp",
        waypoints_xy=np.column_stack([xr_pts, yr_pts]),
        w_left=wl_r,
        w_right=wr_r,
        is_closed=track.is_closed,
        ds_target=ds_target,
    )

    int_k2_base = float(np.sum(track.kappa ** 2) * track.ds)
    int_k2_race = float(np.sum(racing_track.kappa ** 2) * racing_track.ds)
    max_k_base = float(np.max(np.abs(track.kappa)))
    max_k_race = float(np.max(np.abs(racing_track.kappa)))
    within = bool(np.all(best_n >= lb - 1e-6) and np.all(best_n <= ub + 1e-6))
    n_max_phys = float(np.max(np.maximum(np.abs(lb), np.abs(ub))))
    max_abs_n = float(np.max(np.abs(best_n)))
    solve_s = time.perf_counter() - t0

    return RacingLineResult(
        base_track=track,
        racing_track=racing_track,
        n_offset_on_base=best_n,
        lb=lb,
        ub=ub,
        centerline_int_kappa2=int_k2_base,
        racing_int_kappa2=int_k2_race,
        centerline_max_abs_kappa=max_k_base,
        racing_max_abs_kappa=max_k_race,
        within_bounds=within,
        stats={
            "solver": "MT-OCP-BSpline-LBFGSB" if ggv is not None else "MinCurv-FullCorridor",
            "length_base_m": float(track.length),
            "length_racing_m": float(racing_track.length),
            "length_delta_m": float(racing_track.length - track.length),
            "int_k2_reduction_pct": 100.0 * (1.0 - int_k2_race / max(int_k2_base, 1e-9)),
            "max_kappa_reduction_pct": 100.0 * (1.0 - max_k_race / max(max_k_base, 1e-9)),
            "r_min_base_m": 1.0 / max(max_k_base, 1e-6),
            "r_min_racing_m": 1.0 / max(max_k_race, 1e-6),
            "max_abs_n_m": max_abs_n,
            "n_max_phys_m": n_max_phys,
            "corridor_util_pct": 100.0 * max_abs_n / max(n_max_phys, 1e-6),
            "stage_a_qss_s": float(stage_a_lap_s),
            "stage_b_qss_s": float(stage_b_lap_s),
            "opt_iters": float(opt_iters_done),
            "solve_time_s": float(solve_s),
        },
    )


def verify_swept_path_4corners(
    base_track: Track,
    rl: RacingLineResult,
    df_14dof: Any = None,
    w_car: float = 1.42,
    l_f: float = 1.15,
    l_r: float = 0.95,
    safety_margin: float = 0.14,
) -> Dict[str, Any]:
    """
    Exact 4-corner chassis swept-path check against track cone boundaries.
    Checks all 4 corners ((+L_f, +-w/2), (-L_r, +-w/2)) on geometric racing line AND 14-DOF telemetry.
    """
    half_w = 0.5 * w_car
    corners = [(l_f, half_w), (l_f, -half_w), (-l_r, half_w), (-l_r, -half_w)]

    def _corner_lat_bounds(xi_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sin_xi = np.sin(xi_arr)
        cos_xi = np.cos(xi_arr)
        lat_all = [xc * sin_xi + yc * cos_xi for xc, yc in corners]
        return np.maximum.reduce(lat_all), np.minimum.reduce(lat_all)

    _, _, psi_r, _, _ = _eval_curve_fast(base_track, rl.n_offset_on_base)
    xi = np.arctan2(np.sin(psi_r - base_track.psi), np.cos(psi_r - base_track.psi))
    max_lat_g, min_lat_g = _corner_lat_bounds(xi)
    n_max_eff_l = np.maximum(0.15, base_track.w_left - max_lat_g - safety_margin)
    n_max_eff_r = np.maximum(0.15, base_track.w_right + min_lat_g - safety_margin)
    geom_clear_l = base_track.w_left - (rl.n_offset_on_base + max_lat_g)
    geom_clear_r = base_track.w_right + (rl.n_offset_on_base + min_lat_g)
    geom_min_clear = float(np.min(np.minimum(geom_clear_l, geom_clear_r)))
    geom_within_eff = bool(
        np.all(rl.n_offset_on_base <= n_max_eff_l + 1e-4)
        and np.all(rl.n_offset_on_base >= -n_max_eff_r - 1e-4)
    )

    corners_hit = 0
    min_14dof_clear = geom_min_clear
    if df_14dof is not None and len(df_14dof) > 10:
        lap1 = df_14dof[df_14dof["lap"] == 1] if "lap" in df_14dof.columns else df_14dof
        px = lap1["x"].to_numpy(dtype=np.float64)
        py = lap1["y"].to_numpy(dtype=np.float64)
        yaw_rad = np.deg2rad(lap1["yaw_deg"].to_numpy(dtype=np.float64))
        # Nearest centerline point for each telemetry sample
        dx_m = px[:, None] - base_track.x[None, :]
        dy_m = py[:, None] - base_track.y[None, :]
        idx_n = np.argmin(dx_m * dx_m + dy_m * dy_m, axis=1)
        psi_c = base_track.psi[idx_n]
        tx = np.cos(psi_c)
        ty = np.sin(psi_c)
        ey = -(px - base_track.x[idx_n]) * ty + (py - base_track.y[idx_n]) * tx
        xi_t = np.arctan2(np.sin(yaw_rad - psi_c), np.cos(yaw_rad - psi_c))
        max_lat_t, min_lat_t = _corner_lat_bounds(xi_t)
        cl_l = base_track.w_left[idx_n] - (ey + max_lat_t)
        cl_r = base_track.w_right[idx_n] + (ey + min_lat_t)
        cl_min = np.minimum(cl_l, cl_r)
        min_14dof_clear = float(np.min(cl_min))
        hit_mask = cl_min < 0.0
        if np.any(hit_mask):
            hit_bins = set(int(base_track.s[i] // 4.0) for i in idx_n[hit_mask])
            corners_hit = len(hit_bins)

    return {
        "geom_within_eff": geom_within_eff,
        "geom_min_cone_clearance_m": geom_min_clear,
        "min_14dof_cone_clearance_m": min_14dof_clear,
        "corners_cones_hit": int(corners_hit),
    }


def export_enriched_racing_line_json(
    rl: RacingLineResult,
    qss: QSSResult,
    ggv: GGVMap,
    out_path: Path,
    df_14dof: Any = None,
    lap_time_s: Optional[float] = None,
    n_samples: int = 360,
    mech_brake_start: float = 0.35,
) -> Path:
    """
    Export enriched racing line JSON for MirenaSim (RacingLineVisualizer + DilSessionManager).
    Preserves ROS (x, y, s_m, yaw_rad, ax) AND adds (z, s, n, ax_mps2, ay_mps2, throttle, brake,
    brake_regen_frac, brake_mech_frac, steer_rad, kappa, grip_util, is_brake_point, is_apex).
    """
    race = rl.racing_track
    base = rl.base_track
    n_r = race.n_points
    m_out = max(40, min(int(n_samples), n_r))
    s_grid = np.linspace(0.0, float(race.length), m_out, endpoint=False)

    # Interpolate geometry along racing line arc length
    rx = np.interp(s_grid, race.s, race.x)
    ry = np.interp(s_grid, race.s, race.y)
    rpsi = np.unwrap(race.psi)
    psi_g = np.interp(s_grid, race.s, rpsi)
    kap_g = np.interp(s_grid, race.s, race.kappa)

    # Lateral offset n(s) on base track projected at each sample
    s_base_g = np.linspace(0.0, float(base.length), m_out, endpoint=False)
    n_g = np.interp(s_base_g, base.s, rl.n_offset_on_base)

    # Use closed-loop 14-DOF telemetry if provided, otherwise fall back to QSS-GGV profile
    if df_14dof is not None and len(df_14dof) > 20:
        lap1 = df_14dof[df_14dof["lap"] == 1].copy() if "lap" in df_14dof.columns else df_14dof.copy()
        s_raw = lap1["s_m"].to_numpy(dtype=np.float64)
        order = np.argsort(s_raw)
        s_u, uniq = np.unique(s_raw[order], return_index=True)
        idx_u = order[uniq]
        v_g = np.interp(s_grid, s_u, lap1["vx"].to_numpy(dtype=np.float64)[idx_u])
        ax_g = np.interp(s_grid, s_u, lap1["ax"].to_numpy(dtype=np.float64)[idx_u])
        ay_g = np.interp(s_grid, s_u, lap1["ay"].to_numpy(dtype=np.float64)[idx_u])
        thr_g = np.interp(s_grid, s_u, lap1["throttle"].to_numpy(dtype=np.float64)[idx_u])
        brk_g = np.interp(s_grid, s_u, lap1["brake"].to_numpy(dtype=np.float64)[idx_u])
        str_g = np.deg2rad(np.interp(s_grid, s_u, lap1["steer_deg"].to_numpy(dtype=np.float64)[idx_u]))
        t_lap = float(lap_time_s) if lap_time_s is not None else float(lap1["t_s"].iloc[-1] - lap1["t_s"].iloc[0])
    else:
        v_g = np.interp(s_grid, qss.s, qss.v_qss)
        ax_g = np.interp(s_grid, qss.s, qss.ax_qss)
        ay_g = np.interp(s_grid, qss.s, qss.ay_qss)
        thr_g = np.clip(np.where(ax_g >= -0.05, np.maximum(ax_g, 0.0) / 8.5 + 0.12, 0.0), 0.0, 1.0)
        brk_g = np.clip(np.where(ax_g < -0.5, -ax_g / 15.0, 0.0), 0.0, 1.0)
        str_g = np.arctan(1.54 * kap_g)
        t_lap = float(lap_time_s) if lap_time_s is not None else float(qss.lap_time_s)

    # Clean speed-regulation brake micro-pulses & apply zero-phase symmetric smoothing (~0.15 s)
    # Any brake < 0.14 with mild deceleration (ax > -3.5 m/s^2) is classified as coasting/lift.
    brk_g = np.where((brk_g < 0.14) & (ax_g > -3.5), 0.0, brk_g)
    def _circ_smooth(arr: np.ndarray, passes: int = 2) -> np.ndarray:
        out = arr.copy()
        for _ in range(passes):
            out = 0.25 * np.roll(out, 1) + 0.50 * out + 0.25 * np.roll(out, -1)
        return out
    thr_g = np.clip(_circ_smooth(thr_g, passes=2), 0.0, 1.0)
    brk_g = np.clip(_circ_smooth(brk_g, passes=2), 0.0, 1.0)
    brk_g = np.where((brk_g < 0.14) & (ax_g > -3.5), 0.0, brk_g)
    thr_g = np.where(brk_g >= 0.15, 0.0, thr_g)

    ay_lim = np.interp(np.clip(v_g, ggv.speeds[0], ggv.speeds[-1]), ggv.speeds, ggv.ay_max)
    grip_g = np.clip(np.hypot(ax_g, ay_g) / np.maximum(ay_lim, 1.0), 0.0, 1.5)

    # Split brake into regen vs mechanical fractions using TeR27 35/65 blending map
    mbs = float(np.clip(mech_brake_start, 0.05, 0.95))
    reg_frac = np.clip(brk_g / mbs, 0.0, 1.0)
    mech_frac = np.clip((brk_g - mbs) / max(1.0 - mbs, 0.05), 0.0, 1.0)

    # Detect clean Brake Points (real corner-entry braking: brake >= 0.15 & ax <= -3.5, >= 14 m apart),
    # Apexes (local v_min with |kappa| > 0.055), and Exit-to-Full-Gas points (throttle >= 0.65 onset after corner).
    is_bp = np.zeros(m_out, dtype=bool)
    is_apx = np.zeros(m_out, dtype=bool)
    is_exit = np.zeros(m_out, dtype=bool)
    ds_out = float(race.length) / float(m_out)
    min_bp_gap = max(3, int(round(14.0 / max(ds_out, 0.5))))
    min_apx_gap = max(3, int(round(12.0 / max(ds_out, 0.5))))
    last_bp = -min_bp_gap
    for i in range(m_out):
        ip = (i - 1) % m_out
        strong_now = (brk_g[i] >= 0.15 and ax_g[i] <= -3.5)
        strong_prev = (brk_g[ip] >= 0.15 and ax_g[ip] <= -3.5)
        if strong_now and not strong_prev and (i - last_bp) >= min_bp_gap:
            is_bp[i] = True
            last_bp = i

    last_apx = -min_apx_gap
    for i in range(m_out):
        im = (i - 1) % m_out
        ip = (i + 1) % m_out
        im2 = (i - 2) % m_out
        ip2 = (i + 2) % m_out
        if abs(kap_g[i]) > 0.055 and v_g[i] <= v_g[im] + 0.03 and v_g[i] <= v_g[ip] + 0.03 and v_g[i] <= 0.5 * (v_g[im2] + v_g[ip2]) + 0.08:
            if (i - last_apx) >= min_apx_gap:
                is_apx[i] = True
                last_apx = i

    last_exit = -min_apx_gap
    for i in range(m_out):
        ip = (i - 1) % m_out
        if thr_g[i] >= 0.65 and thr_g[ip] < 0.65 and (i - last_exit) >= min_apx_gap:
            is_exit[i] = True
            last_exit = i

    points = []
    for i in range(m_out):
        points.append({
            "x": round(float(rx[i]), 3),
            "y": round(float(ry[i]), 3),
            "z": round(float(-rx[i]), 3),
            "s": round(float(s_grid[i]), 3),
            "s_m": round(float(s_grid[i]), 3),
            "n": round(float(n_g[i]), 3),
            "yaw_rad": round(float(psi_g[i]), 4),
            "v_mps": round(float(v_g[i]), 3),
            "ax": round(float(ax_g[i]), 3),
            "ax_mps2": round(float(ax_g[i]), 3),
            "ay_mps2": round(float(ay_g[i]), 3),
            "throttle": round(float(thr_g[i]), 3),
            "brake": round(float(brk_g[i]), 3),
            "brake_regen_frac": round(float(reg_frac[i]), 3),
            "brake_mech_frac": round(float(mech_frac[i]), 3),
            "steer_rad": round(float(str_g[i]), 4),
            "kappa": round(float(kap_g[i]), 5),
            "grip_util": round(float(grip_g[i]), 3),
            "is_brake_point": bool(is_bp[i]),
            "is_apex": bool(is_apx[i]),
            "is_exit_point": bool(is_exit[i]),
        })

    payload = {
        "track": base.name,
        "solver": str(rl.stats.get("solver", "MT-OCP")),
        "lap_time_s": round(float(t_lap), 3),
        "length_m": round(float(race.length), 2),
        "n_max_phys_m": round(float(rl.stats.get("n_max_phys_m", 0.95)), 3),
        "brake_points_count": int(np.sum(is_bp)),
        "apex_count": int(np.sum(is_apx)),
        "points": points,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

