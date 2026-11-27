"""Vehicle overview router (GET /api/vehicle/overview)."""

from __future__ import annotations

from fastapi import APIRouter

from api.models.vehicle import VehicleOverviewResponse
from api.services.setup_service import build_vehicle_overview

router = APIRouter(prefix="/api/vehicle", tags=["vehicle"])


@router.get("/overview", response_model=VehicleOverviewResponse)
def get_vehicle_overview() -> VehicleOverviewResponse:
    return build_vehicle_overview()
