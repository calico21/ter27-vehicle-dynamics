"""
End-to-End Differentiable Adjoint Optimization Engine for TeR27 (4WD).

Modules:
  - diff_4wd_controller: C2-smooth CasADi 4WD controller (unrolled AL-QP,
    real I^2*R(T) in-hub electrical loss map, aero-pitch torque split,
    smooth latency predictor).
  - discrete_adjoint: Discrete reverse-mode adjoint sensitivities dJ/dtheta
    over full Skidpad, Acceleration (75 m), and Autocross horizons with
    finite-difference verification (< 1e-6).
  - co_optimize_ter27: Joint gradient-based mechanical + control co-optimization
    respecting all physical and FSAE EV constraints.
"""

from vd_correlation_lab.optimization.differentiable_adjoint.co_optimize_ter27 import (
    DEFAULT_REPORT_PATH,
    OPT_CTRL_KEYS,
    OPT_VEH_BOUNDS,
    run_ter27_joint_co_optimization,
)
from vd_correlation_lab.optimization.differentiable_adjoint.diff_4wd_controller import (
    CTRL_PARAM_BOUNDS,
    CTRL_PARAM_NAMES,
    CTRL_PARAM_NOMINAL,
    Diff4WDController,
    smooth_inhub_loss_kw,
    smooth_unrolled_alqp_4wd,
)
from vd_correlation_lab.optimization.differentiable_adjoint.discrete_adjoint import (
    DiscreteAdjointEngine,
)

__all__ = [
    "CTRL_PARAM_BOUNDS",
    "CTRL_PARAM_NAMES",
    "CTRL_PARAM_NOMINAL",
    "DEFAULT_REPORT_PATH",
    "Diff4WDController",
    "DiscreteAdjointEngine",
    "OPT_CTRL_KEYS",
    "OPT_VEH_BOUNDS",
    "run_ter27_joint_co_optimization",
    "smooth_inhub_loss_kw",
    "smooth_unrolled_alqp_4wd",
]
