"""DiL sessions and CasADi reference A/B comparison router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models.telemetry import SessionABComparisonResponse, SessionsListResponse
from api.services.session_service import get_session_comparison, list_sessions_catalog

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=SessionsListResponse)
def list_sessions() -> SessionsListResponse:
    return list_sessions_catalog()


@router.get("/{session_id}", response_model=SessionABComparisonResponse)
def get_session_ab(session_id: str) -> SessionABComparisonResponse:
    result = get_session_comparison(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return result
