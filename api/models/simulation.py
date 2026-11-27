"""Pydantic v2 models for asynchronous CasADi lap-time and dynamic event simulation."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from api.models.tires import GripProfileName, TireCompoundName

SimulationEventName = Literal["skidpad", "accel", "autocross", "endurance"]
SimulationJobStatusName = Literal["queued", "running", "completed", "failed"]
SimulationModeName = Literal["human_achievable", "theoretical_optimal"]


class SimulationRunRequest(BaseModel):
    event: SimulationEventName = Field("skidpad", description="Dynamic event to solve")
    grip_profile: GripProfileName = Field("FSG_RUBBERED", description="Grip profile")
    compound: TireCompoundName = Field("R25B", description="Tire compound")
    optimize_setup: bool = Field(False, description="Enable 7-parameter trackside setup co-optimization")
    mode: SimulationModeName = Field("human_achievable", description="Solver fidelity and driver envelope mode")
    n_intervals: int = Field(120, ge=40, le=500, description="Discretization intervals for full-lap OCP")
    max_iter: int = Field(300, ge=50, le=1200, description="IPOPT maximum iterations")
    dry_run: bool = Field(
        False,
        description="Return cached validated CasADi reference metrics without spawning IPOPT",
    )


class SimulationJobStatusResponse(BaseModel):
    job_id: str
    event: SimulationEventName
    grip_profile: str
    compound: str
    optimize_setup: bool
    status: SimulationJobStatusName
    progress_pct: float = Field(..., ge=0.0, le=100.0)
    created_at_utc: str
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None
    command: List[str]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    log_tail: List[str] = Field(default_factory=list)
