"""Validation Vault catalog router (master_metrics_catalog.json + suites 01..10)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models.validation import VaultCatalogResponse
from api.services.vault_service import load_vault_catalog

router = APIRouter(prefix="/api/vault", tags=["vault"])


@router.get("/catalog", response_model=VaultCatalogResponse)
def get_vault_catalog() -> VaultCatalogResponse:
    catalog = load_vault_catalog()
    if catalog is None:
        raise HTTPException(
            status_code=404,
            detail="Validation vault master_metrics_catalog.json not found",
        )
    return catalog
