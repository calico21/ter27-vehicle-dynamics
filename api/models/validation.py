"""Pydantic v2 models for the Validation Vault catalog (suites 01..11)."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class VaultKpi(BaseModel):
    label: str
    value: str
    status: str


class VaultFigure(BaseModel):
    filename: str
    rel_path: str
    title: str
    size_kb: float
    width_px: int
    height_px: int


class VaultSuite(BaseModel):
    code: str
    folder: str
    title: str
    subtitle: str
    tag: str
    primary_metrics_json: Optional[str] = None
    json_files: List[str]
    csv_files: List[str]
    n_figures: int
    kpis: List[VaultKpi]
    figures: List[VaultFigure]


class VaultCatalogResponse(BaseModel):
    generated_utc: str
    standard: str
    vault_root: str
    n_suites: int
    total_pngs: int
    total_jsons: int
    total_png_size_mb: float
    suites: List[VaultSuite]
