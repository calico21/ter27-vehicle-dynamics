"""Parametric arc-length track representation and cone-track importers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class Track:
    """Uniform arc-length parametric track in 2D ISO-XY plane (+X forward, +Y left)."""

    name: str
    is_closed: bool
    length: float
    ds: float
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    w_left: np.ndarray
    w_right: np.ndarray

    @property
    def n_points(self) -> int:
        return int(len(self.s))

    def left_boundary(self) -> np.ndarray:
        nx = -np.sin(self.psi)
        ny = np.cos(self.psi)
        return np.column_stack([self.x + self.w_left * nx, self.y + self.w_left * ny])

    def right_boundary(self) -> np.ndarray:
        nx = -np.sin(self.psi)
        ny = np.cos(self.psi)
        return np.column_stack([self.x - self.w_right * nx, self.y - self.w_right * ny])

    def check_validity(self, min_clearance_m: Optional[float] = None) -> Dict[str, object]:
        total_width = self.w_left + self.w_right
        min_width = float(np.min(total_width))
        max_kappa = float(np.max(np.abs(self.kappa)))
        min_radius = float(1.0 / max(max_kappa, 1e-6))

        # Closure check
        closure_pos_err = 0.0
        closure_psi_err = 0.0
        closure_kappa_err = 0.0
        if self.is_closed:
            dx_end = (self.x[0] - self.x[-1]) - self.ds * np.cos(self.psi[-1])
            dy_end = (self.y[0] - self.y[-1]) - self.ds * np.sin(self.psi[-1])
            closure_pos_err = float(np.hypot(dx_end, dy_end))
            # Expected psi jump across wrap step
            dpsi_wrap = float(_wrap_angle(self.psi[0] - self.psi[-1]))
            closure_psi_err = float(abs(dpsi_wrap - 0.5 * (self.kappa[0] + self.kappa[-1]) * self.ds))
            closure_kappa_err = float(abs(self.kappa[0] - self.kappa[-1]))

        # Non-adjacent self-clearance check
        pts = np.column_stack([self.x, self.y])
        n = self.n_points
        min_idx_sep = max(12, int(round(12.0 / max(self.ds, 0.1))))
        idx = np.arange(n)
        if self.is_closed:
            sep = np.minimum((idx[:, None] - idx[None, :]) % n, (idx[None, :] - idx[:, None]) % n)
        else:
            sep = np.abs(idx[:, None] - idx[None, :])
        mask = sep > min_idx_sep
        D = cdist(pts, pts)
        min_nonadj_dist = float(np.min(D[mask])) if np.any(mask) else 999.0
        req_clear = float(min_clearance_m if min_clearance_m is not None else np.max(total_width) + 0.5)

        valid = (
            min_width >= 2.999
            and min_nonadj_dist > req_clear
            and (not self.is_closed or closure_pos_err < 0.25 * self.ds)
        )
        return {
            "valid": bool(valid),
            "length_m": float(self.length),
            "n_points": n,
            "ds_m": float(self.ds),
            "min_width_m": min_width,
            "min_radius_m": min_radius,
            "min_nonadj_clearance_m": min_nonadj_dist,
            "closure_pos_err_m": closure_pos_err,
            "closure_psi_err_rad": closure_psi_err,
            "closure_kappa_err_1pm": closure_kappa_err,
        }

    def project_point(self, px: float, py: float, s_hint: Optional[float] = None) -> Tuple[float, float, float]:
        """Return (s_closest, signed_ey_left_pos, track_psi) for inertial point (px, py)."""
        if s_hint is not None and self.n_points > 40:
            i_center = int(round((s_hint % self.length) / self.ds)) % self.n_points
            win = int(round(25.0 / max(self.ds, 0.1)))
            offsets = np.arange(-win, win + 1)
            cand = (i_center + offsets) % self.n_points if self.is_closed else np.clip(i_center + offsets, 0, self.n_points - 1)
            dx = px - self.x[cand]
            dy = py - self.y[cand]
            idx = int(cand[int(np.argmin(dx * dx + dy * dy))])
        else:
            dx = px - self.x
            dy = py - self.y
            idx = int(np.argmin(dx * dx + dy * dy))

        # Refine along local tangent
        tx = np.cos(self.psi[idx])
        ty = np.sin(self.psi[idx])
        dx_i = px - self.x[idx]
        dy_i = py - self.y[idx]
        ds_loc = dx_i * tx + dy_i * ty
        s_cl = self.s[idx] + ds_loc
        if self.is_closed:
            s_cl = float(s_cl % self.length)
        else:
            s_cl = float(np.clip(s_cl, 0.0, self.length))
        ey = -dx_i * ty + dy_i * tx
        return s_cl, float(ey), float(self.psi[idx])

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "is_closed": bool(self.is_closed),
            "length": float(self.length),
            "ds": float(self.ds),
            "s": [round(float(v), 4) for v in self.s],
            "x": [round(float(v), 5) for v in self.x],
            "y": [round(float(v), 5) for v in self.y],
            "psi": [round(float(v), 6) for v in self.psi],
            "kappa": [round(float(v), 6) for v in self.kappa],
            "w_left": [round(float(v), 4) for v in self.w_left],
            "w_right": [round(float(v), 4) for v in self.w_right],
        }

    def save_json(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Track":
        return cls(
            name=str(d["name"]),
            is_closed=bool(d.get("is_closed", True)),
            length=float(d["length"]),
            ds=float(d["ds"]),
            s=np.asarray(d["s"], dtype=np.float64),
            x=np.asarray(d["x"], dtype=np.float64),
            y=np.asarray(d["y"], dtype=np.float64),
            psi=np.asarray(d["psi"], dtype=np.float64),
            kappa=np.asarray(d["kappa"], dtype=np.float64),
            w_left=np.asarray(d["w_left"], dtype=np.float64),
            w_right=np.asarray(d["w_right"], dtype=np.float64),
        )

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "Track":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_track_from_waypoints(
    name: str,
    waypoints_xy: np.ndarray,
    w_left: Union[float, np.ndarray] = 1.80,
    w_right: Union[float, np.ndarray] = 1.80,
    is_closed: bool = True,
    ds_target: float = 0.5,
) -> Track:
    """Interpolate C2 spline from control waypoints and resample to uniform arc length."""
    pts = np.asarray(waypoints_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("waypoints_xy must have shape (N, 2)")

    n_in = len(pts)
    wl_in = np.full(n_in, float(w_left), dtype=np.float64) if np.isscalar(w_left) else np.asarray(w_left, dtype=np.float64)
    wr_in = np.full(n_in, float(w_right), dtype=np.float64) if np.isscalar(w_right) else np.asarray(w_right, dtype=np.float64)

    if is_closed:
        if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
            pts = np.vstack([pts, pts[0]])
            wl_in = np.append(wl_in, wl_in[0])
            wr_in = np.append(wr_in, wr_in[0])
        else:
            pts[-1] = pts[0]
            wl_in[-1] = wl_in[0]
            wr_in[-1] = wr_in[0]
        bc = "periodic"
    else:
        bc = "not-a-knot"

    # Two-pass arc-length parameterization for uniform ds
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t_param = np.concatenate([[0.0], np.cumsum(seg)])
    cs_x = CubicSpline(t_param, pts[:, 0], bc_type=bc)
    cs_y = CubicSpline(t_param, pts[:, 1], bc_type=bc)
    cs_wl = CubicSpline(t_param, wl_in, bc_type=bc)
    cs_wr = CubicSpline(t_param, wr_in, bc_type=bc)

    # Dense evaluation to measure exact arc length
    n_dense = max(2000, int(round(t_param[-1] / 0.05)))
    t_dense = np.linspace(0.0, t_param[-1], n_dense, endpoint=True)
    xd = cs_x(t_dense)
    yd = cs_y(t_dense)
    s_dense = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xd), np.diff(yd)))])
    total_len = float(s_dense[-1])

    # Reparameterize in true arc-length coordinate s -> (x, y)
    cs_sx = CubicSpline(s_dense, xd, bc_type=bc)
    cs_sy = CubicSpline(s_dense, yd, bc_type=bc)
    # Invert t(s) linearly on dense grid for width profiles
    wld = cs_wl(t_dense)
    wrd = cs_wr(t_dense)

    if is_closed:
        n_out = max(40, int(round(total_len / ds_target)))
        ds_actual = total_len / float(n_out)
        s_out = np.arange(n_out, dtype=np.float64) * ds_actual
    else:
        n_out = max(20, int(round(total_len / ds_target)) + 1)
        s_out = np.linspace(0.0, total_len, n_out)
        ds_actual = float(s_out[1] - s_out[0])

    x_out = cs_sx(s_out)
    y_out = cs_sy(s_out)
    dx = cs_sx(s_out, 1)
    dy = cs_sy(s_out, 1)
    ddx = cs_sx(s_out, 2)
    ddy = cs_sy(s_out, 2)
    speed_sq = np.maximum(dx * dx + dy * dy, 1e-12)
    psi_out = np.arctan2(dy, dx)
    kappa_out = (dx * ddy - dy * ddx) / (speed_sq ** 1.5)

    wl_out = np.interp(s_out, s_dense, wld)
    wr_out = np.interp(s_out, s_dense, wrd)

    return Track(
        name=name,
        is_closed=is_closed,
        length=total_len,
        ds=ds_actual,
        s=s_out,
        x=x_out,
        y=y_out,
        psi=psi_out,
        kappa=kappa_out,
        w_left=wl_out,
        w_right=wr_out,
    )


def load_planning_cone_track(
    json_path: Union[str, Path],
    name: Optional[str] = None,
    ds_target: float = 0.5,
) -> Track:
    """Build a closed Track from a Mirena cone JSON file (blue=left, yellow=right)."""
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    cones = data.get("cones", [])
    blue = np.array([[float(c["x"]), float(c["y"])] for c in cones if c.get("type") == "cone_blue"], dtype=np.float64)
    yellow = np.array([[float(c["x"]), float(c["y"])] for c in cones if c.get("type") == "cone_yellow"], dtype=np.float64)
    if len(blue) != len(yellow) or len(blue) < 6:
        raise ValueError(f"Unsupported cone layout in {p}: blue={len(blue)}, yellow={len(yellow)}")
    center = 0.5 * (blue + yellow)
    half_widths = np.maximum(0.5 * np.linalg.norm(blue - yellow, axis=1), 1.50)
    return build_track_from_waypoints(
        name=name or p.stem,
        waypoints_xy=center,
        w_left=half_widths,
        w_right=half_widths,
        is_closed=True,
        ds_target=ds_target,
    )
