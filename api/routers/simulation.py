"""Asynchronous CasADi simulation execution and job status router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models.simulation import SimulationJobStatusResponse, SimulationRunRequest
from api.services.simulation_runner import simulation_runner

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.post("/run", response_model=SimulationJobStatusResponse)
async def start_simulation_job(req: SimulationRunRequest) -> SimulationJobStatusResponse:
    return await simulation_runner.start_job(req)


@router.get("/status/{job_id}", response_model=SimulationJobStatusResponse)
def get_simulation_job_status(job_id: str) -> SimulationJobStatusResponse:
    status = simulation_runner.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Simulation job '{job_id}' not found")
    return status
