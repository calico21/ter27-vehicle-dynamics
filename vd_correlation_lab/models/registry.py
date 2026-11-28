#!/usr/bin/env python3
"""
Unified Python Vehicle Model & Suspension Registry for Tecnun eRacing.

Exposes a clean factory combining all 3 vehicle models orthogonally with
all 4 suspension options without altering full_14dof or fast_3dof:

Vehicle Models:
  1. full_14dof                  - Reference 14-DOF / high-fidelity vehicle model
  2. fast_3dof                   - Fast 3-DOF planar + 4-wheel model (200 Hz, 35 ms LLTD)
  3. differentiable_casadi_14dof - C2-smooth 14-DOF CasADi SX + MF6.2/PINN + C-Codegen

Suspension Modes (orthogonal to vehicle model):
  1. analytic_1d_lut             - 1D camber/toe LUTs + constant MR + digressive dampers + ARB
  2. 3d_double_wishbone_lut      - 3D double-wishbone kinematics + progressive MR/RC/SVSA
  3. decoupled_heave_roll        - Decoupled 3rd-element heave + roll stiffness/damping
  4. standard_pushrod            - Classic corner-coilover pushrod + U-bar ARB
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vd_correlation_lab.models.differentiable_casadi.suspension_casadi import (
    SUSPENSION_MODES,
    normalize_suspension_mode,
)
from vd_correlation_lab.models.differentiable_casadi.vehicle_14dof_casadi import (
    DifferentiableVehicle14Dof,
)
from vd_correlation_lab.models.vehicle_14dof.ter27_physics import (
    TeR27PhysicsModel,
)

MODEL_FULL_14DOF: str = "full_14dof"
MODEL_FAST_3DOF: str = "fast_3dof"
MODEL_DIFFERENTIABLE_CASADI_14DOF: str = "differentiable_casadi_14dof"

VEHICLE_MODELS: Tuple[str, ...] = (
    MODEL_FULL_14DOF,
    MODEL_FAST_3DOF,
    MODEL_DIFFERENTIABLE_CASADI_14DOF,
)


def list_vehicle_models() -> List[Dict[str, Any]]:
    """Return metadata for all 3 registered vehicle dynamics models."""
    return [
        {
            "id": MODEL_FULL_14DOF,
            "name": "Full 14-DOF",
            "dof": 14,
            "recommended_hz": 1000,
            "differentiable": False,
            "supports_codegen": False,
            "description": "Full 6-DOF chassis + 4 unsprung suspensions + 4 wheels with dynamic load transfer.",
        },
        {
            "id": MODEL_FAST_3DOF,
            "name": "Fast 3-DOF",
            "dof": 7,
            "recommended_hz": 200,
            "differentiable": False,
            "supports_codegen": False,
            "description": "Planar 3-DOF chassis + 4 wheels with quasi-static 35 ms LLTD filter for Driverless.",
        },
        {
            "id": MODEL_DIFFERENTIABLE_CASADI_14DOF,
            "name": "Differentiable CasADi 14-DOF",
            "dof": 14,
            "recommended_hz": 200,
            "differentiable": True,
            "supports_codegen": True,
            "description": "C2-smooth 14-DOF CasADi SX model with MF6.2+PINN tires, carcass relaxation, exact AD Jacobians, and C-Codegen.",
        },
    ]


def list_suspension_modes() -> List[Dict[str, Any]]:
    """Return metadata for all orthogonal suspension modes."""
    return [
        {
            "id": "analytic_1d_lut",
            "name": "Analytic 1D LUT",
            "description": "1D camber/toe LUTs from suspension.json with constant motion ratio, digressive dampers, bumpstops, and ARB.",
        },
        {
            "id": "3d_double_wishbone_lut",
            "name": "3D Double Wishbone LUT",
            "description": "3D double-wishbone spatial kinematics with progressive motion ratio, roll center, bump-steer, and SVSA anti-pitch.",
        },
        {
            "id": "decoupled_heave_roll",
            "name": "Decoupled Heave-Roll",
            "description": "Decoupled 3rd-element heave spring/damper plus independent roll stiffness and roll damping.",
        },
        {
            "id": "standard_pushrod",
            "name": "Standard Pushrod",
            "description": "Classic corner-coilover pushrod suspension with linear asymmetric dampers and U-bar ARB.",
        },
    ]


def normalize_model_id(model_id: str) -> str:
    """Normalize a vehicle model identifier or alias to its canonical ID."""
    clean = str(model_id).strip().lower().replace("-", "_")
    aliases = {
        "full_14dof": MODEL_FULL_14DOF,
        "14dof": MODEL_FULL_14DOF,
        "full": MODEL_FULL_14DOF,
        "fast_3dof": MODEL_FAST_3DOF,
        "3dof": MODEL_FAST_3DOF,
        "fast": MODEL_FAST_3DOF,
        "differentiable_casadi_14dof": MODEL_DIFFERENTIABLE_CASADI_14DOF,
        "differentiable_casadi": MODEL_DIFFERENTIABLE_CASADI_14DOF,
        "diff_14dof": MODEL_DIFFERENTIABLE_CASADI_14DOF,
        "casadi_14dof": MODEL_DIFFERENTIABLE_CASADI_14DOF,
        "option3": MODEL_DIFFERENTIABLE_CASADI_14DOF,
    }
    if clean not in aliases:
        raise ValueError(f"Unknown vehicle model '{model_id}'. Valid models: {VEHICLE_MODELS}")
    return aliases[clean]


@dataclass
class RegisteredModelHandle:
    """Uniform wrapper returned for legacy full_14dof and fast_3dof selections."""

    model_id: str
    vehicle_id: str
    suspension_mode: str
    grip_profile: str
    backend: Any
    tau_wt_s: float = 0.0

    @property
    def differentiable(self) -> bool:
        return self.model_id == MODEL_DIFFERENTIABLE_CASADI_14DOF


def create_vehicle_model(
    model_id: str = MODEL_DIFFERENTIABLE_CASADI_14DOF,
    vehicle_id: str = "ter27_4wd",
    suspension_mode: str = "3d_double_wishbone_lut",
    grip_profile: str = "FSG_RUBBERED",
    tire_compound: Optional[str] = None,
    use_pinn: bool = True,
    dt: float = 0.005,
    rk4_substeps: int = 2,
    use_thermal_coupling: bool = False,
) -> Any:
    """
    Factory creating any of the 3 vehicle models combined orthogonally with any
    of the 4 suspension modes.

    Returns:
      - DifferentiableVehicle14Dof when model_id == 'differentiable_casadi_14dof'
      - RegisteredModelHandle wrapping the untouched reference model for
        'full_14dof' and 'fast_3dof'
    """
    norm_model = normalize_model_id(model_id)
    norm_susp = normalize_suspension_mode(suspension_mode)

    if norm_model == MODEL_DIFFERENTIABLE_CASADI_14DOF:
        model = DifferentiableVehicle14Dof(
            vehicle_id=vehicle_id,
            suspension_mode=norm_susp,
            tire_compound=tire_compound,
            use_pinn=use_pinn,
            grip_profile=grip_profile,
            dt=dt,
            rk4_substeps=rk4_substeps,
            use_thermal_coupling=use_thermal_coupling,
        )
        model.model_id = MODEL_DIFFERENTIABLE_CASADI_14DOF
        model.suspension_mode = norm_susp
        return model

    vid = str(vehicle_id).strip().lower()
    if vid in ("ter26", "ter26_rwd", "rwd"):
        from vd_correlation_lab.models.vehicle_14dof.ter26_high_fidelity_model import (
            TeR26HighFidelityCasADi,
        )
        use_3d = norm_susp == "3d_double_wishbone_lut"
        backend = TeR26HighFidelityCasADi(use_3d_kinematics=use_3d)
        vid = "ter26_rwd"
    else:
        backend = TeR27PhysicsModel(profile=grip_profile)
        vid = "ter27_4wd"

    tau_wt = 0.035 if norm_model == MODEL_FAST_3DOF else 0.0
    return RegisteredModelHandle(
        model_id=norm_model,
        vehicle_id=vid,
        suspension_mode=norm_susp,
        grip_profile=grip_profile,
        backend=backend,
        tau_wt_s=tau_wt,
    )
