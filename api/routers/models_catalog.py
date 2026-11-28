"""Models catalog router (GET /api/models/catalog)."""

from __future__ import annotations

from fastapi import APIRouter

from api.models.vehicle import ModelsCatalogResponse
from api.services.setup_service import build_models_catalog

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/catalog", response_model=ModelsCatalogResponse)
def get_models_catalog() -> ModelsCatalogResponse:
    return build_models_catalog()
