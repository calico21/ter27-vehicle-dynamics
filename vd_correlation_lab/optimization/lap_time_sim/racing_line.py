"""Minimum-curvature racing line optimizer within strict FS track-width bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.sparse import diags, lil_matrix

from .track import Track, build_track_from_waypoints


@dataclass
class RacingLineResult:
    """Optimal racing line track plus lateral offset and curvature telemetry."""

    base_track: Track
    racing_track: Track
    n_offset_on_base: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    centerline_int_kappa2: float
    racing_int_kappa2: float
    centerline_max_abs_kappa: float
    racing_max_abs_kappa: float
    within_bounds: bool
    stats: Dict[str, float]


def _build_second_deriv_matrix(n: int, ds: float, is_closed: bool):
    inv_ds2 = 1.0 / max(ds * ds, 1e-8)
    D2 = lil_matrix((n, n), dtype=np.float64)
    for i in range(n):
        if is_closed:
            im1 = (i - 1) % n
            ip1 = (i + 1) % n
            D2[i, im1] = inv_ds2
            D2[i, i] = -2.0 * inv_ds2
            D2[i, ip1] = inv_ds2
        else:
            if 0 < i < n - 1:
                D2[i, i - 1] = inv_ds2
                D2[i, i] = -2.0 * inv_ds2
                D2[i, i + 1] = inv_ds2
            elif i == 0:
                D2[i, 0] = 1.0 * inv_ds2
                D2[i, 1] = -2.0 * inv_ds2
                D2[i, 2] = 1.0 * inv_ds2
            else:
                D2[i, n - 3] = 1.0 * inv_ds2
                D2[i, n - 2] = -2.0 * inv_ds2
                D2[i, n - 1] = 1.0 * inv_ds2
    return D2.tocsc()


def _evaluate_exact_curvature(
    track: Track,
    n_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact (x_r, y_r, kappa_r, seg_len) along displaced curve r = r_c + n * m."""
    nx = -np.sin(track.psi)
    ny = np.cos(track.psi)
    xr = track.x + n_vec * nx
    yr = track.y + n_vec * ny
    if track.is_closed:
        xw = np.append(xr, xr[0])
        yw = np.append(yr, yr[0])
        seg = np.hypot(np.diff(xw), np.diff(yw))
        t = np.concatenate([[0.0], np.cumsum(seg)])
        csx = CubicSpline(t, xw, bc_type="periodic")
        csy = CubicSpline(t, yw, bc_type="periodic")
        te = t[:-1]
    else:
        seg = np.hypot(np.diff(xr), np.diff(yr))
        t = np.concatenate([[0.0], np.cumsum(seg)])
        csx = CubicSpline(t, xr, bc_type="not-a-knot")
        csy = CubicSpline(t, yr, bc_type="not-a-knot")
        te = t
    dx = csx(te, 1)
    dy = csy(te, 1)
    ddx = csx(te, 2)
    ddy = csy(te, 2)
    sp2 = np.maximum(dx * dx + dy * dy, 1e-12)
    kr = (dx * ddy - dy * ddx) / (sp2 ** 1.5)
    return xr, yr, kr, seg


def optimize_racing_line(
    track: Track,
    car_track_width_m: float = 1.230,
    safety_margin_m: float = 0.35,
    smooth_reg: float = 0.08,
    n_sqp_iters: int = 5,
    ds_target: float = 0.5,
) -> RacingLineResult:
    """
    Solve bounded minimum-curvature racing line problem via sequential linearized QP (L-BFGS-B):
        min_n  sum_i kappa_r,i(n)^2 * ds_i + smooth_reg * sum_i (n''_i)^2 * ds
        s.t.   -(w_R - car_track/2 - margin) <= n_i <= (w_L - car_track/2 - margin)
    """
    n = track.n_points
    half_car = 0.5 * car_track_width_m + safety_margin_m
    ub = np.maximum(track.w_left - half_car, 0.0)
    lb = -np.maximum(track.w_right - half_car, 0.0)

    for i in range(n):
        if track.kappa[i] > 1e-4:
            ub[i] = min(ub[i], 0.75 / track.kappa[i])
        elif track.kappa[i] < -1e-4:
            lb[i] = max(lb[i], 0.75 / track.kappa[i])

    D2 = _build_second_deriv_matrix(n, track.ds, track.is_closed)
    k_c = track.kappa
    ds0 = track.ds

    n_cur = np.zeros(n, dtype=np.float64)
    _, _, k_cur, seg_cur = _evaluate_exact_curvature(track, n_cur)
    best_n = n_cur.copy()
    best_obj = float(np.sum(k_cur * k_cur * seg_cur[:n]))

    for _it in range(n_sqp_iters):
        # Local Frenet Jacobian: d(kappa_r)/dn = (D2 + k_c^2) / (1 - k_c * n)^2
        denom = np.maximum(1.0 - k_c * n_cur, 0.25)
        inv_d2 = diags(1.0 / (denom * denom), 0, format="csc")
        K2_diag = diags(k_c * k_c, 0, format="csc")
        J_k = inv_d2 @ (D2 + K2_diag)
        J_kT = J_k.T.tocsc()
        D2T = D2.T.tocsc()
        w_s = seg_cur[:n]

        bounds = list(zip(lb - n_cur, ub - n_cur))

        def qp_obj_grad(dn: np.ndarray) -> tuple[float, np.ndarray]:
            k_pred = k_cur + J_k @ dn
            n_tot = n_cur + dn
            d2n = D2 @ n_tot
            obj = 0.5 * float(np.sum(w_s * k_pred * k_pred)) + 0.5 * smooth_reg * float(np.sum(d2n * d2n) * ds0)
            grad = J_kT @ (w_s * k_pred) + (smooth_reg * ds0) * (D2T @ d2n)
            return obj, grad

        res = minimize(
            qp_obj_grad,
            x0=np.zeros(n, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 180, "ftol": 1e-10, "gtol": 1e-7},
        )
        dn = res.x

        # Line search on true nonlinear spline curvature energy
        step = 1.0
        accepted = False
        cur_tot = best_obj + 0.5 * smooth_reg * float(np.sum((D2 @ n_cur) ** 2) * ds0)
        for _ in range(6):
            n_try = np.clip(n_cur + step * dn, lb, ub)
            _, _, k_try, seg_try = _evaluate_exact_curvature(track, n_try)
            obj_k = float(np.sum(k_try * k_try * seg_try[:n]))
            obj_tot = obj_k + smooth_reg * float(np.sum((D2 @ n_try) ** 2) * ds0)
            if obj_tot < cur_tot - 1e-9:
                n_cur = n_try
                k_cur = k_try
                seg_cur = seg_try
                best_n = n_try.copy()
                best_obj = obj_k
                accepted = True
                break
            step *= 0.5
        if not accepted or np.linalg.norm(step * dn, ord=np.inf) < 1e-4:
            break

    nx = -np.sin(track.psi)
    ny = np.cos(track.psi)
    xr_pts = track.x + best_n * nx
    yr_pts = track.y + best_n * ny

    wl_r = np.maximum(track.w_left - best_n, 0.5 * car_track_width_m)
    wr_r = np.maximum(track.w_right + best_n, 0.5 * car_track_width_m)

    racing_track = build_track_from_waypoints(
        name=f"{track.name}_racing_line",
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
            "length_base_m": float(track.length),
            "length_racing_m": float(racing_track.length),
            "length_delta_m": float(racing_track.length - track.length),
            "int_k2_reduction_pct": 100.0 * (1.0 - int_k2_race / max(int_k2_base, 1e-9)),
            "max_kappa_reduction_pct": 100.0 * (1.0 - max_k_race / max(max_k_base, 1e-9)),
            "r_min_base_m": 1.0 / max(max_k_base, 1e-6),
            "r_min_racing_m": 1.0 / max(max_k_race, 1e-6),
            "max_abs_n_m": float(np.max(np.abs(best_n))),
        },
    )
