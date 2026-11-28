"""
Trackside Race Engineering Engine ("15 Minutes Between Runs").

Provides:
- log_ingestor: CAN (.log/.asc/.blf), CSV, Parquet, and trace ingestion with
  RL <-> FR tire temp remap, ISO 8855 sign enforcement, and maneuver segmentation.
- fast_adjoint_sysid: <30s Gauss-Newton / adjoint SysID on libter_diff_14dof.so
  for TeR26 (RWD) and TeR27 (4WD, 280.0 kg).
- setup_advisor: <15 min pit-box setup recommendations with FSAE EV invariant
  verification.
"""

from vd_correlation_lab.trackside.fast_adjoint_sysid import (
    OPT_PARAM_NAMES,
    SysIdResult,
    run_fast_adjoint_sysid,
)
from vd_correlation_lab.trackside.log_ingestor import (
    IngestedRun,
    ingest_log,
    ingest_trace,
    remap_tire_temperatures_rl_fr,
    segment_run,
)
from vd_correlation_lab.trackside.setup_advisor import (
    EXCLUDED_HARDWARE_SWAPS,
    PitBoxAdjustment,
    SetupAdvisorResult,
    advise_pit_box_setup,
    verify_fsae_ev_invariants,
)

__all__ = [
    "IngestedRun",
    "ingest_log",
    "ingest_trace",
    "remap_tire_temperatures_rl_fr",
    "segment_run",
    "OPT_PARAM_NAMES",
    "SysIdResult",
    "run_fast_adjoint_sysid",
    "EXCLUDED_HARDWARE_SWAPS",
    "PitBoxAdjustment",
    "SetupAdvisorResult",
    "advise_pit_box_setup",
    "verify_fsae_ev_invariants",
]
