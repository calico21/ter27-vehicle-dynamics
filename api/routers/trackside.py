"""Trackside race engineering router (POST /api/trackside/analyze)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models.trackside import TracksideAnalyzeRequest, TracksideAnalyzeResponse
from api.services.trackside_service import analyze_trackside_run

router = APIRouter(prefix="/api/trackside", tags=["trackside"])


@router.post("/analyze", response_model=TracksideAnalyzeResponse)
def post_trackside_analyze(req: TracksideAnalyzeRequest) -> TracksideAnalyzeResponse:
    try:
        return analyze_trackside_run(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
