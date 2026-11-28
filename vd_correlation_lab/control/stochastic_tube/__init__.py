"""
Stochastic Tube-MPC (Chance-Constrained Control) package for TeR27 4WD.
"""

from vd_correlation_lab.control.stochastic_tube.chance_constraints import (
    Z_QUANTILE_1E4,
    ChanceBackoffResult,
    ChanceConstraintConfig,
    ChanceConstraintEvaluator,
    norm_ppf_approx,
)
from vd_correlation_lab.control.stochastic_tube.covariance_propagator import (
    EmbeddedDiagonalTubePropagator,
    ExactCovariancePropagator,
    ProcessNoiseConfig,
    ProcessNoiseModel,
)
from vd_correlation_lab.control.stochastic_tube.tube_mpc_controller import (
    StochasticTubeMPCController,
    TubeMPCConfig,
)

__all__ = [
    "Z_QUANTILE_1E4",
    "ChanceBackoffResult",
    "ChanceConstraintConfig",
    "ChanceConstraintEvaluator",
    "norm_ppf_approx",
    "ProcessNoiseConfig",
    "ProcessNoiseModel",
    "ExactCovariancePropagator",
    "EmbeddedDiagonalTubePropagator",
    "TubeMPCConfig",
    "StochasticTubeMPCController",
]
