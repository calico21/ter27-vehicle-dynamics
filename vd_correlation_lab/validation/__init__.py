"""Validation suite for differentiable CasADi 14-DOF vehicle dynamics models."""

from vd_correlation_lab.validation.differentiable_ter26_correlation import (
    DEFAULT_REPORT_PATH,
    run_full_ter26_validation,
    verify_adjoint_gradients_vs_fd,
    run_adjoint_sysid_recovery,
)

__all__ = [
    "DEFAULT_REPORT_PATH",
    "run_full_ter26_validation",
    "verify_adjoint_gradients_vs_fd",
    "run_adjoint_sysid_recovery",
]
