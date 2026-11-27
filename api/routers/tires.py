"""Parametric Hoosier 16.0x7.5-10 Pacejka MF 6.2 tire evaluation router."""

from __future__ import annotations

from fastapi import APIRouter

from api.models.tires import TireEvaluateRequest, TireEvaluateResponse
from api.services.tire_service import evaluate_tire_curves

router = APIRouter(prefix="/api/tires", tags=["tires"])


@router.post("/evaluate", response_model=TireEvaluateResponse)
def evaluate_tire(req: TireEvaluateRequest) -> TireEvaluateResponse:
    return evaluate_tire_curves(req)
