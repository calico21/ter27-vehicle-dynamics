"""Service for loading the Validation Vault master metrics catalog."""

from __future__ import annotations

import json
from typing import List, Optional

from api.core.config import VAULT_CATALOG_PATH
from api.models.validation import VaultCatalogResponse, VaultFigure, VaultKpi, VaultSuite


def load_vault_catalog() -> Optional[VaultCatalogResponse]:
    if not VAULT_CATALOG_PATH.exists():
        return None

    raw = json.loads(VAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    summary = raw.get("summary", {})
    suites_out: List[VaultSuite] = []

    for s in raw.get("suites", []):
        kpis = [
            VaultKpi(
                label=str(k.get("label", "")),
                value=str(k.get("value", "")),
                status=str(k.get("status", "pass")),
            )
            for k in s.get("kpis", [])
        ]
        figs = [
            VaultFigure(
                filename=str(f.get("filename", "")),
                rel_path=str(f.get("rel_path", "")),
                title=str(f.get("title", "")),
                size_kb=float(f.get("size_kb", 0.0)),
                width_px=int(f.get("width_px", 0)),
                height_px=int(f.get("height_px", 0)),
            )
            for f in s.get("figures", [])
        ]
        suites_out.append(
            VaultSuite(
                code=str(s.get("code", "")),
                folder=str(s.get("folder", "")),
                title=str(s.get("title", "")),
                subtitle=str(s.get("subtitle", "")),
                tag=str(s.get("tag", "")),
                primary_metrics_json=s.get("primary_metrics_json"),
                json_files=[str(x) for x in s.get("json_files", [])],
                csv_files=[str(x) for x in s.get("csv_files", [])],
                n_figures=int(s.get("n_figures", len(figs))),
                kpis=kpis,
                figures=figs,
            )
        )

    return VaultCatalogResponse(
        generated_utc=str(raw.get("generated_utc", "")),
        standard=str(raw.get("standard", "")),
        vault_root="outputs/validation_vault",
        n_suites=int(summary.get("n_suites", len(suites_out))),
        total_pngs=int(summary.get("total_pngs", 0)),
        total_jsons=int(summary.get("total_jsons", 0)),
        total_png_size_mb=float(summary.get("total_png_size_mb", 0.0)),
        suites=suites_out,
    )
