"""Fast (< 50 ms) deterministic Stage-1 Quasi-Steady-State (QSS) Forward-Backward Lap Solver."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ggv_extractor import GGVMap
from .track import Track


@dataclass
class QSSResult:
    """Output trajectory, accelerations, lap time, and energy budget from QSS."""

    track_name: str
    controller_label: str
    s: np.ndarray
    ds_seg: np.ndarray
    v_apex: np.ndarray
    v_qss: np.ndarray
    ax_qss: np.ndarray
    ay_qss: np.ndarray
    p_dc_kw: np.ndarray
    dt_seg: np.ndarray
    lap_time_s: float
    v_max_kmh: float
    v_min_kmh: float
    e_net_kj: float
    e_regen_kj: float
    solve_time_ms: float


class _FastGGVLookup:
    """Zero-overhead scalar bilinear lookup for inner QSS loop (< 15 ms per 2200-node lap)."""

    def __init__(self, ggv: GGVMap) -> None:
        self.sp = ggv.speeds.astype(np.float64).tolist()
        self.af = ggv.ay_fracs.astype(np.float64).tolist()
        self.ay_m = ggv.ay_max.astype(np.float64).tolist()
        self.ax_p = ggv.ax_max.astype(np.float64).tolist()
        self.ax_n = ggv.ax_min.astype(np.float64).tolist()
        self.p_acc = ggv.p_acc_kw.astype(np.float64).tolist()
        self.p_brk = ggv.p_brk_kw.astype(np.float64).tolist()
        self.nv = len(self.sp)
        self.nf = len(self.af)
        self.cd_a = float(ggv.cd_a)
        self.mass = max(float(ggv.mass_kg), 100.0)
        self.v0 = self.sp[0]
        self.v1 = self.sp[-1]
        diffs = np.diff(ggv.speeds)
        self.uniform_v = bool(np.allclose(diffs, diffs[0], atol=1e-6))
        self.inv_dv = (1.0 / float(diffs[0])) if self.uniform_v else 0.0

    def ay_max(self, v: float) -> float:
        if v <= self.v0:
            return self.ay_m[0]
        if v >= self.v1:
            return self.ay_m[-1]
        if self.uniform_v:
            f = (v - self.v0) * self.inv_dv
            i = int(f)
            if i >= self.nv - 1:
                return self.ay_m[-1]
            u = f - i
            return self.ay_m[i] * (1.0 - u) + self.ay_m[i + 1] * u
        for i in range(self.nv - 1):
            if v <= self.sp[i + 1]:
                u = (v - self.sp[i]) / max(self.sp[i + 1] - self.sp[i], 1e-6)
                return self.ay_m[i] * (1.0 - u) + self.ay_m[i + 1] * u
        return self.ay_m[-1]

    def _interp2(self, tbl: list, v: float, frac: float) -> float:
        # Speed bracket
        if v <= self.v0:
            iv = 0
            uv = 0.0
        elif v >= self.v1:
            iv = self.nv - 2
            uv = 1.0
        elif self.uniform_v:
            fv = (v - self.v0) * self.inv_dv
            iv = min(max(int(fv), 0), self.nv - 2)
            uv = fv - iv
        else:
            iv = 0
            while iv < self.nv - 2 and v > self.sp[iv + 1]:
                iv += 1
            uv = (v - self.sp[iv]) / max(self.sp[iv + 1] - self.sp[iv], 1e-6)

        # Lateral fraction bracket (non-uniform: [0, 0.35, 0.65, 0.85, 0.96, 1.0])
        if frac <= 0.0:
            jf = 0
            uf = 0.0
        elif frac >= 1.0:
            jf = self.nf - 2
            uf = 1.0
        else:
            jf = 0
            while jf < self.nf - 2 and frac > self.af[jf + 1]:
                jf += 1
            uf = (frac - self.af[jf]) / max(self.af[jf + 1] - self.af[jf], 1e-6)

        row0 = tbl[iv]
        row1 = tbl[iv + 1]
        c00 = row0[jf]
        c10 = row1[jf]
        c01 = row0[jf + 1]
        c11 = row1[jf + 1]
        r0 = c00 * (1.0 - uv) + c10 * uv
        r1 = c01 * (1.0 - uv) + c11 * uv
        return r0 * (1.0 - uf) + r1 * uf

    def limits(self, v: float, ay_abs: float) -> tuple[float, float, float]:
        ay_lim = max(self.ay_max(v), 0.5)
        frac = min(max(ay_abs / ay_lim, 0.0), 1.0)
        ax_hi = self._interp2(self.ax_p, v, frac)
        ax_lo = self._interp2(self.ax_n, v, frac)
        if v > self.v1:
            p30 = max(self._interp2(self.p_acc, self.v1, frac), 0.0)
            df_drag = 0.5 * 1.225 * self.cd_a * (v * v - self.v1 * self.v1)
            f30 = p30 * 900.0 / max(self.v1, 1.0)
            fv = p30 * 900.0 / max(v, 1.0)
            ax_hi = max(ax_hi + (fv - f30 - df_drag) / self.mass, -2.0)
            ax_lo = min(ax_lo - df_drag / self.mass, -0.5)
        return ay_lim, ax_hi, ax_lo

    def est_power_kw(self, v: float, ax: float, ay_abs: float) -> float:
        ay_lim = max(self.ay_max(v), 0.5)
        frac = min(max(ay_abs / ay_lim, 0.0), 1.0)
        if ax >= 0.0:
            ax_top = max(self._interp2(self.ax_p, v, frac), 0.2)
            ratio = min(max(ax / ax_top, 0.0), 1.0)
            p_max = self._interp2(self.p_acc, v, frac)
            p_cruise = (0.5 * 1.225 * self.cd_a * v * v + 0.015 * self.mass * 9.81) * v / 900.0
            return max(p_cruise, ratio * p_max)
        ax_bot = min(self._interp2(self.ax_n, v, frac), -0.5)
        ratio = min(max(abs(ax) / abs(ax_bot), 0.0), 1.0)
        p_reg = self._interp2(self.p_brk, v, frac)
        return min(0.0, max(p_reg, ratio * 1.4 * p_reg))


def solve_qss(
    track: Track,
    ggv: GGVMap,
    v_top_limit_ms: float = 37.4,
    grip_scale: float = 1.0,
    v0_open_ms: Optional[float] = None,
) -> QSSResult:
    """
    Deterministic QSS Apex -> Forward -> Backward solver over arc-length track.
    Runs 2 periodic wrap passes when track.is_closed is True.
    """
    t0 = time.perf_counter()
    fast = _FastGGVLookup(ggv)
    n = track.n_points
    kappa_abs = np.abs(track.kappa)

    # 1. Compute corner speed limit v_apex(s) via fixed-point iteration on ay_max(v)
    ay_grid_eval = np.maximum(fast.ay_m[0], 1.0) * grip_scale
    v_apex = np.full(n, v_top_limit_ms, dtype=np.float64)
    curv_mask = kappa_abs > 1.0e-5
    if np.any(curv_mask):
        v_est = np.sqrt(ay_grid_eval / kappa_abs[curv_mask])
        for _ in range(4):
            ay_v = np.interp(np.clip(v_est, fast.v0, fast.v1), fast.sp, fast.ay_m) * grip_scale
            v_est = np.sqrt(np.maximum(ay_v / kappa_abs[curv_mask], 9.0))
        v_apex[curv_mask] = np.minimum(v_est, v_top_limit_ms)

    # Segment lengths (length N for closed loop, N-1 for open)
    if track.is_closed:
        dx_s = np.roll(track.x, -1) - track.x
        dy_s = np.roll(track.y, -1) - track.y
        ds_seg = np.hypot(dx_s, dy_s)
    else:
        ds_seg = np.hypot(np.diff(track.x), np.diff(track.y))

    v_list = v_apex.tolist()
    if not track.is_closed and v0_open_ms is not None:
        v_list[0] = min(v_list[0], float(v0_open_ms))
    vap_list = v_apex.tolist()
    kap_list = kappa_abs.tolist()
    ds_list = ds_seg.tolist()

    # 2. Forward acceleration passes (2 passes for periodic boundary convergence)
    n_passes = 2 if track.is_closed else 1
    for _p in range(n_passes):
        if track.is_closed:
            for k in range(n):
                vk = v_list[k]
                ayk = vk * vk * kap_list[k]
                _, ax_hi, _ = fast.limits(vk, ayk)
                ax_hi *= grip_scale
                kn = k + 1 if k + 1 < n else 0
                v2_next = vk * vk + 2.0 * ax_hi * ds_list[k]
                v_next = math.sqrt(v2_next if v2_next > 9.0 else 9.0)
                if v_next < v_list[kn]:
                    v_list[kn] = v_next
                else:
                    va = vap_list[kn]
                    v_list[kn] = va if va < v_next else v_next
        else:
            for k in range(n - 1):
                vk = v_list[k]
                ayk = vk * vk * kap_list[k]
                _, ax_hi, _ = fast.limits(vk, ayk)
                ax_hi *= grip_scale
                v2_next = vk * vk + 2.0 * ax_hi * ds_list[k]
                v_next = math.sqrt(v2_next if v2_next > 9.0 else 9.0)
                va = vap_list[k + 1]
                v_list[k + 1] = va if va < v_next else v_next

    for k in range(n):
        if v_list[k] > vap_list[k]:
            v_list[k] = vap_list[k]

    # 3. Backward braking passes (2 passes for periodic closure)
    for _p in range(n_passes):
        if track.is_closed:
            for k in range(n - 1, -1, -1):
                kn = k + 1 if k + 1 < n else 0
                v_kn = v_list[kn]
                ay_kn = v_kn * v_kn * kap_list[kn]
                _, _, ax_lo = fast.limits(v_kn, ay_kn)
                ax_lo *= grip_scale
                v2_prev = v_kn * v_kn - 2.0 * ax_lo * ds_list[k]
                v_prev_lim = math.sqrt(v2_prev if v2_prev > 9.0 else 9.0)
                if v_prev_lim < v_list[k]:
                    v_list[k] = v_prev_lim
        else:
            for k in range(n - 2, -1, -1):
                v_kn = v_list[k + 1]
                ay_kn = v_kn * v_kn * kap_list[k + 1]
                _, _, ax_lo = fast.limits(v_kn, ay_kn)
                ax_lo *= grip_scale
                v2_prev = v_kn * v_kn - 2.0 * ax_lo * ds_list[k]
                v_prev_lim = math.sqrt(v2_prev if v2_prev > 9.0 else 9.0)
                if v_prev_lim < v_list[k]:
                    v_list[k] = v_prev_lim

    v = np.asarray(v_list, dtype=np.float64)

    # 4. Kinematics, lap time, and DC power/energy estimate
    if track.is_closed:
        v_next_arr = np.roll(v, -1)
        ax_qss = (v_next_arr * v_next_arr - v * v) / (2.0 * np.maximum(ds_seg, 1e-4))
        dt_seg = 2.0 * ds_seg / np.maximum(v + v_next_arr, 1.0)
    else:
        v_next_arr = v[1:]
        ax_qss = np.concatenate([
            (v_next_arr * v_next_arr - v[:-1] * v[:-1]) / (2.0 * np.maximum(ds_seg, 1e-4)),
            np.array([0.0]),
        ])
        dt_seg = 2.0 * ds_seg / np.maximum(v[:-1] + v_next_arr, 1.0)

    ay_qss = v * v * track.kappa
    p_dc = np.zeros(n, dtype=np.float64)
    for k in range(n):
        p_dc[k] = fast.est_power_kw(float(v[k]), float(ax_qss[k]), float(abs(ay_qss[k])))

    if track.is_closed:
        e_net_kj = float(np.sum(p_dc * dt_seg))
        e_regen_kj = float(np.sum(-np.minimum(p_dc, 0.0) * dt_seg))
    else:
        e_net_kj = float(np.sum(0.5 * (p_dc[:-1] + p_dc[1:]) * dt_seg))
        p_mid = 0.5 * (p_dc[:-1] + p_dc[1:])
        e_regen_kj = float(np.sum(-np.minimum(p_mid, 0.0) * dt_seg))

    solve_ms = (time.perf_counter() - t0) * 1000.0
    return QSSResult(
        track_name=track.name,
        controller_label=ggv.label,
        s=track.s.copy(),
        ds_seg=ds_seg,
        v_apex=v_apex,
        v_qss=v,
        ax_qss=ax_qss,
        ay_qss=ay_qss,
        p_dc_kw=p_dc,
        dt_seg=dt_seg,
        lap_time_s=float(np.sum(dt_seg)),
        v_max_kmh=float(np.max(v) * 3.6),
        v_min_kmh=float(np.min(v) * 3.6),
        e_net_kj=e_net_kj,
        e_regen_kj=e_regen_kj,
        solve_time_ms=solve_ms,
    )
