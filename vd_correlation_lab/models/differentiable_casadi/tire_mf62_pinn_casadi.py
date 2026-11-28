#!/usr/bin/env python3
"""
Symbolic C2-smooth Pacejka MF 6.2 + Physics-Informed Residual Neural Network (PINN)
tire model in CasADi SX for Hoosier 16.0x7.5-10 R20 and R25B.

Evaluates real calibrated MF 6.2 coefficients and real 4->16->16->2 Tanh PINN
weights with C-infinity physics gates directly inside the differentiable CasADi
SX expression graph (100% C-Codegen compatible).

Coordinate convention: ISO 8855 / ROS REP-103 (+x forward, +y left, +z up).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import casadi as ca
import numpy as np

from vd_correlation_lab.models.differentiable_casadi.smooth_ops import (
    is_casadi,
    sabs,
    sclip,
    smax,
    smin,
    spos,
    ssign,
    smooth_step_01,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MF62_DIR = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Dynamics" / "Tires" / "MF62"
NEURAL_DIR = REPO_ROOT / "src" / "mirena_sim" / "Scenes" / "Vehicle" / "Dynamics" / "Tires" / "NeuralCorrection"

# Standardized track grip targets (preserved intact)
COLD_TEST_OCT: float = 0.47
FSG_RUBBERED: float = 0.57


class TireMF62PINNCasADi:
    """
    Symbolic C2-smooth MF 6.2 + Residual PINN tire evaluator in CasADi SX.

    Supports both Hoosier 16.0x7.5-10 R20 and R25B compounds.
    All operations are pure algebraic/trigonometric/tanh expressions in SX,
    enabling exact 1st/2nd-order AD and GCC -O3 C-Codegen.
    """

    def __init__(
        self,
        compound: str = "R20",
        use_pinn: bool = True,
        mf62_dir: Path = MF62_DIR,
        neural_dir: Path = NEURAL_DIR,
    ) -> None:
        self.compound = str(compound).upper()
        self.use_pinn = bool(use_pinn)

        # Load calibrated MF 6.2 JSON
        mf_path = mf62_dir / f"hoosier_16x7.5_10_{self.compound.lower()}_mf62.json"
        if not mf_path.exists():
            mf_path = mf62_dir / "hoosier_16x7.5_10_r20_mf62.json"
        self.mf_path = mf_path
        self.d: Dict[str, Any] = json.loads(mf_path.read_text(encoding="utf-8"))

        self.fnomin: float = float(self.d["FNOMIN"])
        self.pinom: float = float(self.d["PINOM"])
        self.r0: float = float(self.d["UNLOADED_RADIUS"])
        self.dfz_max: float = (
            (float(self.d["FZ_VALID_MAX"]) - self.fnomin) / self.fnomin
            if "FZ_VALID_MAX" in self.d
            else float("inf")
        )
        for k, v in self.d.items():
            if isinstance(v, (int, float)):
                setattr(self, k, float(v))

        # Load real PINN residual weights JSON
        nn_path = neural_dir / f"neural_residual_weights_{self.compound.lower()}.json"
        if not nn_path.exists():
            nn_path = neural_dir / "neural_residual_weights.json"
        self.nn_path = nn_path
        self.nn_data: Dict[str, Any] = json.loads(nn_path.read_text(encoding="utf-8"))

        norm = self.nn_data["normalization"]
        w = self.nn_data["weights"]
        gates = self.nn_data.get("physics_gates", {})
        arch = self.nn_data.get("architecture", {})

        self.x_mean = np.asarray(norm["x_mean"], dtype=np.float64)
        self.x_std = np.asarray(norm["x_std"], dtype=np.float64)
        self.y_mean = np.asarray(norm["y_mean"], dtype=np.float64)
        self.y_std = np.asarray(norm["y_std"], dtype=np.float64)

        self.w1 = np.asarray(w["w1"], dtype=np.float64)  # (16, 4)
        self.b1 = np.asarray(w["b1"], dtype=np.float64)  # (16,)
        self.w2 = np.asarray(w["w2"], dtype=np.float64)  # (16, 16)
        self.b2 = np.asarray(w["b2"], dtype=np.float64)  # (16,)
        self.w3 = np.asarray(w["w3"], dtype=np.float64)  # (2, 16)
        self.b3 = np.asarray(w["b3"], dtype=np.float64)  # (2,)

        self.mirror_for_right_side: bool = bool(arch.get("mirror_for_right_side", True))
        self.gate_slip_scale: float = float(gates.get("slip_scale_rad", 0.006))
        self.gate_load_trans: float = float(gates.get("load_trans_n", 100.0))
        self.gate_alpha_limit: float = float(gates.get("alpha_limit_rad", 0.22))
        self.gate_alpha_slope: float = float(gates.get("alpha_slope_rad", 0.02))
        self.gate_fz_limit: float = float(gates.get("fz_limit_n", 1300.0))
        self.gate_fz_slope: float = float(gates.get("fz_slope_n", 100.0))
        self.gate_clip_fy: float = float(gates.get("clip_fy_n", 200.0))
        self.gate_clip_mz: float = float(gates.get("clip_mz_nm", 10.0))

        # Calibrated C2 smoothing constants (< 1e-4 relative error vs exact)
        self.smoothing: Dict[str, float] = {
            "eps_fz_n": 0.1,
            "eps_dfz": 1e-4,
            "eps_gamma_rad": 1e-5,
            "eps_camber_f": 1e-4,
            "eps_e_clip": 1e-4,
            "eps_g_floor": 1e-5,
            "eps_sign_slip": 1e-4,
            "eps_alpha_eq_sq": 1e-10,
            "eps_gate_t": 1e-4,
        }

    @staticmethod
    def _shape_cos(b: Any, c: Any, e: Any, x: Any) -> Any:
        bx = b * x
        if is_casadi(bx):
            return ca.cos(c * ca.atan(bx - e * (bx - ca.atan(bx))))
        return math.cos(c * math.atan(bx - e * (bx - math.atan(bx))))

    def eval_pinn_symbolic(
        self,
        alpha: Any,
        fz: Any,
        gamma: Any,
        p_kpa: Any,
        side: float = 1.0,
    ) -> Tuple[Any, Any]:
        """
        Evaluate the 4->16->16->2 Tanh residual PINN with smooth C2 physics gates
        in CasADi SX/MX (or float).

        Args:
            alpha: Slip angle [rad]
            fz: Normal load [N]
            gamma: Inclination/camber angle [rad]
            p_kpa: Inflation pressure [kPa]
            side: +1.0 for left wheel (FL, RL), -1.0 for right wheel (FR, RR)

        Returns:
            (delta_fy_N, delta_mz_Nm)
        """
        sgn = float(side) if self.mirror_for_right_side else 1.0
        a_in = sgn * alpha
        g_in = sgn * gamma

        x_vec = ca.vertcat(a_in, fz, g_in, p_kpa)
        x_norm = (x_vec - ca.DM(self.x_mean)) / ca.DM(self.x_std)

        h1 = ca.tanh(ca.mtimes(ca.DM(self.w1), x_norm) + ca.DM(self.b1))
        h2 = ca.tanh(ca.mtimes(ca.DM(self.w2), h1) + ca.DM(self.b2))
        out_raw = ca.mtimes(ca.DM(self.w3), h2) + ca.DM(self.b3)

        raw_fy = out_raw[0] * self.y_std[0] + self.y_mean[0]
        raw_mz = out_raw[1] * self.y_std[1] + self.y_mean[1]

        # Smooth C-infinity physics gates matching TireModelEvaluator.eval_pinn
        g_slip = 1.0 - ca.exp(-(a_in * a_in + g_in * g_in) / (2.0 * self.gate_slip_scale ** 2))
        g_load = smooth_step_01(fz / self.gate_load_trans, eps=self.smoothing["eps_gate_t"])
        a_abs = ca.sqrt(a_in * a_in + 1e-6)
        # Clamp sigmoid exponents softly to prevent exp overflow at extreme inputs
        exp_a = sclip((a_abs - self.gate_alpha_limit) / self.gate_alpha_slope, -40.0, 40.0, 1e-2)
        exp_fz = sclip((fz - self.gate_fz_limit) / self.gate_fz_slope, -40.0, 40.0, 1e-2)
        g_alpha = 1.0 / (1.0 + ca.exp(exp_a))
        g_fz = 1.0 / (1.0 + ca.exp(exp_fz))
        scale = g_slip * g_load * g_alpha * g_fz

        delta_fy = sgn * self.gate_clip_fy * ca.tanh(raw_fy * scale / self.gate_clip_fy)
        delta_mz = sgn * self.gate_clip_mz * ca.tanh(raw_mz * scale / self.gate_clip_mz)
        return delta_fy, delta_mz

    def eval_combined_symbolic(
        self,
        kappa: Any,
        alpha: Any,
        fz: Any,
        gamma: Any,
        side: float,
        lmux: Any,
        lmuy: Any,
        p_kpa: Any = 83.0,
        kya_scale: Any = 1.0,
        camber_scale: Any = 1.0,
        pinn_scale: Any = 1.0,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Evaluate C2-smooth MF 6.2 + optional PINN residual combined-slip forces/moments.

        Returns:
            (Fx [N], Fy [N], Mz [Nm], mu_y_eff [-], Gyk [-])
        """
        s = self.smoothing

        fz_pos = spos(fz, s["eps_fz_n"])
        dfz_raw = (fz_pos - self.fnomin) / self.fnomin
        dfz = smin(dfz_raw, self.dfz_max, s["eps_dfz"]) if math.isfinite(self.dfz_max) else dfz_raw
        dpi = (p_kpa - self.pinom) / self.pinom

        gamma_eff = gamma * camber_scale
        abs_gamma = sabs(gamma_eff, s["eps_gamma_rad"])
        gamma_sq = gamma_eff * gamma_eff

        # --- Pure lateral Fy0 ---
        cy = self.PCY1
        camber_f_y = smax(1.0 - self.PDY3 * gamma_sq, 0.2, s["eps_camber_f"])
        mu_y = (
            (self.PDY1 + self.PDY2 * dfz)
            * (1.0 + self.PPY3 * dpi + self.PPY4 * dpi * dpi)
            * camber_f_y
            * lmuy
        )
        dy = mu_y * fz_pos
        shy = side * (self.PHY1 + self.PHY2 * dfz) + self.PHY3 * gamma_eff
        svy = side * fz_pos * (self.PVY1 + self.PVY2 * dfz) + fz_pos * (self.PVY3 + self.PVY4 * dfz) * gamma_eff
        alpha_y = alpha + shy
        kya = (
            kya_scale
            * self.PKY1
            * self.fnomin
            * (1.0 + self.PPY1 * dpi)
            * (1.0 - self.PKY3 * abs_gamma)
            * ca.sin(2.0 * ca.atan(fz_pos / (self.PKY2 * self.fnomin * (1.0 + self.PPY2 * dpi) + 1e-4)))
        )
        by = kya / (cy * dy + 1e-4)
        ey_raw = (self.PEY1 + self.PEY2 * dfz) * (
            1.0 + self.PEY5 * gamma_sq - (self.PEY3 + self.PEY4 * gamma_eff) * ssign(alpha_y, s["eps_sign_slip"])
        )
        ey = sclip(ey_raw, -10.0, 1.0, s["eps_e_clip"])
        y_arg = by * alpha_y
        fy0 = dy * ca.sin(cy * ca.atan(y_arg - ey * (y_arg - ca.atan(y_arg)))) + svy

        # --- Pure longitudinal Fx0 ---
        cx = self.PCX1
        camber_f_x = smax(1.0 - self.PDX3 * gamma_sq, 0.2, s["eps_camber_f"])
        mu_x = (
            (self.PDX1 + self.PDX2 * dfz)
            * (1.0 + self.PPX3 * dpi + self.PPX4 * dpi * dpi)
            * camber_f_x
            * lmux
        )
        dx = mu_x * fz_pos
        kx = kappa + self.PHX1 + self.PHX2 * dfz
        svx = fz_pos * (self.PVX1 + self.PVX2 * dfz)
        kxk = (
            fz_pos
            * (self.PKX1 + self.PKX2 * dfz)
            * ca.exp(self.PKX3 * dfz)
            * (1.0 + self.PPX1 * dpi + self.PPX2 * dpi * dpi)
        )
        bx = kxk / (cx * dx + 1e-4)
        ex_raw = (self.PEX1 + self.PEX2 * dfz + self.PEX3 * dfz * dfz) * (
            1.0 - self.PEX4 * ssign(kx, s["eps_sign_slip"])
        )
        ex = sclip(ex_raw, -10.0, 1.0, s["eps_e_clip"])
        x_arg = bx * kx
        fx0 = dx * ca.sin(cx * ca.atan(x_arg - ex * (x_arg - ca.atan(x_arg)))) + svx

        # --- Combined slip weighting ---
        bxa = (self.RBX1 + self.RBX3 * gamma_sq) * ca.cos(ca.atan(self.RBX2 * kappa))
        exa = sclip(self.REX1 + self.REX2 * dfz, -10.0, 1.0, s["eps_e_clip"])
        gxa_den = smax(self._shape_cos(bxa, self.RCX1, exa, self.RHX1), 1e-4, 1e-5)
        gxa = smax(self._shape_cos(bxa, self.RCX1, exa, alpha + self.RHX1) / gxa_den, 0.0, s["eps_g_floor"])

        byk = (self.RBY1 + self.RBY4 * gamma_sq) * ca.cos(ca.atan(self.RBY2 * (alpha - self.RBY3)))
        eyk = sclip(self.REY1 + self.REY2 * dfz, -10.0, 1.0, s["eps_e_clip"])
        shyk = self.RHY1 + self.RHY2 * dfz
        gyk_den = smax(self._shape_cos(byk, self.RCY1, eyk, shyk), 1e-4, 1e-5)
        gyk = smax(self._shape_cos(byk, self.RCY1, eyk, kappa + shyk) / gyk_den, 0.0, s["eps_g_floor"])

        dvyk = (
            mu_y
            * fz_pos
            * (self.RVY1 + self.RVY2 * dfz + self.RVY3 * gamma_eff)
            * ca.cos(ca.atan(self.RVY4 * alpha))
        )
        svyk = dvyk * ca.sin(self.RVY5 * ca.atan(self.RVY6 * kappa))

        fx = gxa * fx0
        fy_mf = gyk * fy0 + svyk

        # --- Self-aligning torque Mz ---
        shf = shy + svy / (kya + 1e-3)
        sht = self.QHZ1 + self.QHZ2 * dfz + (self.QHZ3 + self.QHZ4 * dfz) * gamma_eff
        alpha_t = alpha + sht
        alpha_r = alpha + shf
        kx_eq = (kxk / (kya + 1e-3)) * kappa
        alpha_t_mag = ca.sqrt(alpha_t * alpha_t + kx_eq * kx_eq + s["eps_alpha_eq_sq"])
        alpha_r_mag = ca.sqrt(alpha_r * alpha_r + kx_eq * kx_eq + s["eps_alpha_eq_sq"])
        alpha_t_signed = ssign(alpha_t, s["eps_sign_slip"]) * alpha_t_mag

        dt = (
            (fz_pos / self.fnomin)
            * self.r0
            * (self.QDZ1 + self.QDZ2 * dfz)
            * (1.0 + self.PPZ1 * dpi)
            * (1.0 + self.QDZ3 * abs_gamma + self.QDZ4 * gamma_sq)
        )
        ct = self.QCZ1
        bt = (self.QBZ1 + self.QBZ2 * dfz + self.QBZ3 * dfz * dfz) * (
            1.0 + self.QBZ4 * gamma_eff + self.QBZ5 * abs_gamma
        )
        t_arg = bt * alpha_t_mag
        et_raw = (self.QEZ1 + self.QEZ2 * dfz + self.QEZ3 * dfz * dfz) * (
            1.0 + (self.QEZ4 + self.QEZ5 * gamma_eff) * (2.0 / math.pi) * ca.atan(bt * ct * alpha_t_signed)
        )
        et = sclip(et_raw, -10.0, 1.0, s["eps_e_clip"])
        trail = dt * ca.cos(ct * ca.atan(t_arg - et * (t_arg - ca.atan(t_arg)))) * ca.cos(alpha)
        mz_prime = -trail * (fy_mf - svyk)

        dr = (
            fz_pos
            * self.r0
            * (side * (self.QDZ6 + self.QDZ7 * dfz) + (self.QDZ8 + self.QDZ9 * dfz) * (1.0 + self.PPZ2 * dpi) * gamma_eff)
            * ca.cos(alpha)
        )
        br = self.QBZ9 + self.QBZ10 * by * cy
        mzr = dr * ca.cos(ca.atan(br * alpha_r_mag))

        s_arm = self.r0 * (self.SSZ1 + self.SSZ2 * (fy_mf / self.fnomin) + (self.SSZ3 + self.SSZ4 * dfz) * gamma_eff)
        mz_mf = mz_prime + mzr + s_arm * fx

        # --- Hybrid PINN residual addition (scaled by combined-slip Gyk and lmuy) ---
        if self.use_pinn:
            dfy_nn, dmz_nn = self.eval_pinn_symbolic(alpha, fz_pos, gamma_eff, p_kpa, side=side)
            fy = fy_mf + pinn_scale * gyk * lmuy * dfy_nn
            mz = mz_mf + pinn_scale * gyk * lmuy * dmz_nn
        else:
            fy = fy_mf
            mz = mz_mf

        return fx, fy, mz, mu_y, gyk

    def build_casadi_function(self, name: str = "tire_mf62_pinn") -> ca.Function:
        """Build a standalone CasADi SX Function for single-wheel evaluation."""
        kappa = ca.SX.sym("kappa")
        alpha = ca.SX.sym("alpha")
        fz = ca.SX.sym("fz")
        gamma = ca.SX.sym("gamma")
        lmux = ca.SX.sym("lmux")
        lmuy = ca.SX.sym("lmuy")
        p_kpa = ca.SX.sym("p_kpa")
        fx, fy, mz, muy, gyk = self.eval_combined_symbolic(
            kappa, alpha, fz, gamma, side=1.0, lmux=lmux, lmuy=lmuy, p_kpa=p_kpa
        )
        return ca.Function(
            name,
            [kappa, alpha, fz, gamma, lmux, lmuy, p_kpa],
            [fx, fy, mz, muy, gyk],
            ["kappa", "alpha", "fz", "gamma", "lmux", "lmuy", "p_kpa"],
            ["fx", "fy", "mz", "mu_y", "gyk"],
        )
