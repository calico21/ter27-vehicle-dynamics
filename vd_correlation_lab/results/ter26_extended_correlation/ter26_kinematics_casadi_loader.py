"""
CasADi C^2 Cubic B-Spline Kinematics Loader for TeR26.

Loads exact 3D double-wishbone kinematics lookup tables from `ter26_kinematics_lut.npz`
and exposes C^2-smooth `casadi.interpolant(..., 'bspline', ...)` functions for use in
CasADi + IPOPT Multiple-Shooting SysID and Optimal Control models.

Units in symbolic API:
  - Inputs: z_w_m in [-0.040, +0.040] m (bump > 0), d_rack_m in [-0.045, +0.045] m (+Y = left)
  - Outputs: SI units (rad, m, rad/m, dimensionless) for *_rad / *_m methods,
    and deg / mm for *_deg / *_mm methods.
  - Vehicle conventions:
      * Camber < 0: top of tire leans inward toward vehicle centerline (both L and R).
      * Steer > 0: wheel turns left (ISO +Z CCW from above) for both FL and FR.
      * Toe > 0: toe-in (front of tire points inward toward centerline).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Union

import casadi as ca
import numpy as np

Scalar = Union[float, ca.SX, ca.MX, ca.DM]

DEFAULT_NPZ_PATH = Path(__file__).resolve().parent / "ter26_kinematics_lut.npz"


class TeR26KinematicsCasADi:
    """C^2 CasADi cubic B-Spline surrogate of exact TeR26 3D double-wishbone kinematics."""

    def __init__(self, npz_path: Union[str, Path] = DEFAULT_NPZ_PATH) -> None:
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=False)

        self.z_grid_m: np.ndarray = np.asarray(data["z_grid_m"], dtype=np.float64)
        self.rack_grid_m: np.ndarray = np.asarray(data["rack_grid_m"], dtype=np.float64)
        self.mean_steer_grid_rad: np.ndarray = np.asarray(data["mean_steer_grid_rad"], dtype=np.float64)
        self.metadata: Dict[str, Any] = json.loads(str(data["metadata_json"]))
        self.static_summary: Dict[str, Any] = self.metadata["static_summary"]

        grid_f = [self.z_grid_m.tolist(), self.rack_grid_m.tolist()]
        grid_r = [self.z_grid_m.tolist()]
        grid_s = [self.mean_steer_grid_rad.tolist()]

        def _bspline_2d(key: str) -> ca.Function:
            arr = np.asarray(data[key], dtype=np.float64)
            # CasADi 2D interpolant expects Fortran-order flattening (grid_0 fastest)
            return ca.interpolant(f"ter26_{key}", "bspline", grid_f, arr.ravel(order="F"))

        def _bspline_1d(key: str, grid: list) -> ca.Function:
            arr = np.asarray(data[key], dtype=np.float64)
            return ca.interpolant(f"ter26_{key}", "bspline", grid, arr.ravel())

        # Front Left 2D cubic B-Splines: f(z_w_m, d_rack_m)
        # Note: Front Right is exact mirror: FR(z_w, d_rack) has
        #   camber_FR(z_w, d_rack)      =  camber_FL(z_w, -d_rack)
        #   steer_FR(z_w, d_rack)       = -steer_FL(z_w, -d_rack)
        #   toe_FR(z_w, d_rack)         =  toe_FL(z_w, -d_rack)
        #   damper_disp_FR(z_w, d_rack) =  damper_disp_FL(z_w, -d_rack)
        #   mr_spring_FR(z_w, d_rack)   =  mr_spring_FL(z_w, -d_rack)
        #   arb_angle_FR(z_w, d_rack)   =  arb_angle_FL(z_w, -d_rack)
        #   mr_arb_FR(z_w, d_rack)      =  mr_arb_FL(z_w, -d_rack)
        #   rc_z_FR(z_w, d_rack)        =  rc_z_FL(z_w, -d_rack)
        #   svsa_slope_FR(z_w, d_rack)  =  svsa_slope_FL(z_w, -d_rack)
        #   trail_FR(z_w, d_rack)       =  trail_FL(z_w, -d_rack)
        #   scrub_FR(z_w, d_rack)       =  scrub_FL(z_w, -d_rack)
        self._fl_camber_deg = _bspline_2d("fl_camber_deg")
        self._fl_steer_deg = _bspline_2d("fl_steer_deg")
        self._fl_toe_deg = _bspline_2d("fl_toe_deg")
        self._fl_damper_disp_mm = _bspline_2d("fl_damper_disp_mm")
        self._fl_mr_spring = _bspline_2d("fl_mr_spring")
        self._fl_arb_angle_rad = _bspline_2d("fl_arb_angle_rad")
        self._fl_mr_arb_rad_m = _bspline_2d("fl_mr_arb_rad_m")
        self._fl_rc_z_mm = _bspline_2d("fl_rc_z_mm")
        self._fl_svsa_slope = _bspline_2d("fl_svsa_slope")
        self._fl_mech_trail_mm = _bspline_2d("fl_mech_trail_mm")
        self._fl_scrub_radius_mm = _bspline_2d("fl_scrub_radius_mm")
        self._fl_cp_dx_mm = _bspline_2d("fl_cp_dx_mm")
        self._fl_cp_dy_mm = _bspline_2d("fl_cp_dy_mm")

        # Rear Left 1D cubic B-Splines: f(z_w_m) (RR is identical in vehicle conventions)
        self._rl_camber_deg = _bspline_1d("rl_camber_deg", grid_r)
        self._rl_toe_deg = _bspline_1d("rl_toe_deg", grid_r)
        self._rl_damper_disp_mm = _bspline_1d("rl_damper_disp_mm", grid_r)
        self._rl_mr_spring = _bspline_1d("rl_mr_spring", grid_r)
        self._rl_arb_angle_rad = _bspline_1d("rl_arb_angle_rad", grid_r)
        self._rl_mr_arb_rad_m = _bspline_1d("rl_mr_arb_rad_m", grid_r)
        self._rl_rc_z_mm = _bspline_1d("rl_rc_z_mm", grid_r)
        self._rl_svsa_slope = _bspline_1d("rl_svsa_slope", grid_r)
        self._rl_cp_dx_mm = _bspline_1d("rl_cp_dx_mm", grid_r)
        self._rl_cp_dy_mm = _bspline_1d("rl_cp_dy_mm", grid_r)

        # Calibrated 1D B-Spline: mean front steer angle (rad) -> rack displacement (m)
        self._rack_from_mean_steer_m = _bspline_1d("rack_from_mean_steer_m", grid_s)

        self._deg2rad = math.pi / 180.0
        self._mm2m = 1e-3

    # -------------------------------------------------------------------------
    # Steering Rack Mapping
    # -------------------------------------------------------------------------

    def rack_from_mean_steer_m(self, mean_steer_rad: Scalar) -> Scalar:
        """Return steering rack lateral displacement (m) for desired mean front steer (rad)."""
        return self._rack_from_mean_steer_m(mean_steer_rad)

    def rack_from_steering_wheel_rad(self, sw_angle_rad: Scalar) -> Scalar:
        """Convert steering wheel angle (rad, + = left) to rack displacement (m)."""
        ratio_m_per_rad = (self.metadata["params"]["steering_ratio_mm_per_rev"] * 1e-3) / (2.0 * math.pi)
        return sw_angle_rad * ratio_m_per_rad

    # -------------------------------------------------------------------------
    # Front Corner Evaluators (SI: rad, m, rad/m)
    # -------------------------------------------------------------------------

    def fl_camber_deg(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_camber_deg(ca.vertcat(z_w_m, d_rack_m))

    def fr_camber_deg(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_camber_deg(ca.vertcat(z_w_m, -d_rack_m))

    def fl_camber_rad(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self.fl_camber_deg(z_w_m, d_rack_m) * self._deg2rad

    def fr_camber_rad(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self.fr_camber_deg(z_w_m, d_rack_m) * self._deg2rad

    def fl_steer_deg(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_steer_deg(ca.vertcat(z_w_m, d_rack_m))

    def fr_steer_deg(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return -self._fl_steer_deg(ca.vertcat(z_w_m, -d_rack_m))

    def fl_steer_rad(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self.fl_steer_deg(z_w_m, d_rack_m) * self._deg2rad

    def fr_steer_rad(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self.fr_steer_deg(z_w_m, d_rack_m) * self._deg2rad

    def fl_toe_deg(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_toe_deg(ca.vertcat(z_w_m, d_rack_m))

    def fr_toe_deg(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_toe_deg(ca.vertcat(z_w_m, -d_rack_m))

    def fl_damper_disp_mm(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_damper_disp_mm(ca.vertcat(z_w_m, d_rack_m))

    def fr_damper_disp_mm(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_damper_disp_mm(ca.vertcat(z_w_m, -d_rack_m))

    def fl_damper_disp_m(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self.fl_damper_disp_mm(z_w_m, d_rack_m) * self._mm2m

    def fr_damper_disp_m(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self.fr_damper_disp_mm(z_w_m, d_rack_m) * self._mm2m

    def fl_mr_spring(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_mr_spring(ca.vertcat(z_w_m, d_rack_m))

    def fr_mr_spring(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_mr_spring(ca.vertcat(z_w_m, -d_rack_m))

    def fl_arb_angle_rad(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_arb_angle_rad(ca.vertcat(z_w_m, d_rack_m))

    def fr_arb_angle_rad(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_arb_angle_rad(ca.vertcat(z_w_m, -d_rack_m))

    def fl_mr_arb_rad_m(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_mr_arb_rad_m(ca.vertcat(z_w_m, d_rack_m))

    def fr_mr_arb_rad_m(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_mr_arb_rad_m(ca.vertcat(z_w_m, -d_rack_m))

    def fl_rc_z_mm(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_rc_z_mm(ca.vertcat(z_w_m, d_rack_m))

    def fr_rc_z_mm(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_rc_z_mm(ca.vertcat(z_w_m, -d_rack_m))

    def fl_rc_z_m(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self.fl_rc_z_mm(z_w_m, d_rack_m) * self._mm2m

    def fr_rc_z_m(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self.fr_rc_z_mm(z_w_m, d_rack_m) * self._mm2m

    def fl_svsa_slope(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_svsa_slope(ca.vertcat(z_w_m, d_rack_m))

    def fl_mech_trail_mm(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_mech_trail_mm(ca.vertcat(z_w_m, d_rack_m))

    def fl_scrub_radius_mm(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_scrub_radius_mm(ca.vertcat(z_w_m, d_rack_m))

    def fl_cp_dx_mm(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_cp_dx_mm(ca.vertcat(z_w_m, d_rack_m))

    def fl_cp_dy_mm(self, z_w_m: Scalar, d_rack_m: Scalar) -> Scalar:
        return self._fl_cp_dy_mm(ca.vertcat(z_w_m, d_rack_m))

    # -------------------------------------------------------------------------
    # Rear Corner Evaluators (1D in z_w_m; RL and RR identical in vehicle conv.)
    # -------------------------------------------------------------------------

    def rear_camber_deg(self, z_w_m: Scalar) -> Scalar:
        return self._rl_camber_deg(z_w_m)

    def rear_camber_rad(self, z_w_m: Scalar) -> Scalar:
        return self._rl_camber_deg(z_w_m) * self._deg2rad

    def rear_toe_deg(self, z_w_m: Scalar) -> Scalar:
        return self._rl_toe_deg(z_w_m)

    def rear_toe_rad(self, z_w_m: Scalar) -> Scalar:
        return self._rl_toe_deg(z_w_m) * self._deg2rad

    def rear_damper_disp_mm(self, z_w_m: Scalar) -> Scalar:
        return self._rl_damper_disp_mm(z_w_m)

    def rear_damper_disp_m(self, z_w_m: Scalar) -> Scalar:
        return self._rl_damper_disp_mm(z_w_m) * self._mm2m

    def rear_mr_spring(self, z_w_m: Scalar) -> Scalar:
        return self._rl_mr_spring(z_w_m)

    def rear_arb_angle_rad(self, z_w_m: Scalar) -> Scalar:
        return self._rl_arb_angle_rad(z_w_m)

    def rear_mr_arb_rad_m(self, z_w_m: Scalar) -> Scalar:
        return self._rl_mr_arb_rad_m(z_w_m)

    def rear_rc_z_mm(self, z_w_m: Scalar) -> Scalar:
        return self._rl_rc_z_mm(z_w_m)

    def rear_rc_z_m(self, z_w_m: Scalar) -> Scalar:
        return self._rl_rc_z_mm(z_w_m) * self._mm2m

    def rear_svsa_slope(self, z_w_m: Scalar) -> Scalar:
        return self._rl_svsa_slope(z_w_m)

    def rear_cp_dx_mm(self, z_w_m: Scalar) -> Scalar:
        return self._rl_cp_dx_mm(z_w_m)

    def rear_cp_dy_mm(self, z_w_m: Scalar) -> Scalar:
        return self._rl_cp_dy_mm(z_w_m)

    # -------------------------------------------------------------------------
    # Convenience Axle Evaluators for Vehicle Dynamics Models
    # -------------------------------------------------------------------------

    def eval_front_axle_from_mean_steer(
        self,
        z_fl_m: Scalar,
        z_fr_m: Scalar,
        mean_steer_rad: Scalar,
    ) -> Dict[str, Scalar]:
        """Evaluate FL & FR steer, camber, MRs, and RC height from mean steer (rad)."""
        d_rack = self.rack_from_mean_steer_m(mean_steer_rad)
        return {
            "d_rack_m": d_rack,
            "delta_fl_rad": self.fl_steer_rad(z_fl_m, d_rack),
            "delta_fr_rad": self.fr_steer_rad(z_fr_m, d_rack),
            "gamma_fl_rad": self.fl_camber_rad(z_fl_m, d_rack),
            "gamma_fr_rad": self.fr_camber_rad(z_fr_m, d_rack),
            "mr_spring_fl": self.fl_mr_spring(z_fl_m, d_rack),
            "mr_spring_fr": self.fr_mr_spring(z_fr_m, d_rack),
            "mr_arb_fl_rad_m": self.fl_mr_arb_rad_m(z_fl_m, d_rack),
            "mr_arb_fr_rad_m": self.fr_mr_arb_rad_m(z_fr_m, d_rack),
            "h_rc_f_m": 0.5 * (self.fl_rc_z_m(z_fl_m, d_rack) + self.fr_rc_z_m(z_fr_m, d_rack)),
        }


def load_ter26_kinematics(npz_path: Union[str, Path] = DEFAULT_NPZ_PATH) -> TeR26KinematicsCasADi:
    """Convenience factory returning an initialized TeR26KinematicsCasADi instance."""
    return TeR26KinematicsCasADi(npz_path=npz_path)
