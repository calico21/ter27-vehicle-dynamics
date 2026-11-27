"""Trackside setup comparison matrix and live MirenaSim UDP apply router."""

from __future__ import annotations

from fastapi import APIRouter

from api.models.setups import (
    ApplySetupRequest,
    ApplySetupResponse,
    SetupsMatrixResponse,
)
from api.services.setup_service import apply_trackside_setup, build_setups_matrix

router = APIRouter(prefix="/api/setups", tags=["setups"])


@router.get("", response_model=SetupsMatrixResponse)
def get_setups_matrix() -> SetupsMatrixResponse:
    return build_setups_matrix()


@router.post("/apply", response_model=ApplySetupResponse)
def apply_setup(req: ApplySetupRequest) -> ApplySetupResponse:
    return apply_trackside_setup(req)
