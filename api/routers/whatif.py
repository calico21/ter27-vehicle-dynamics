"""What-If simulation sweep router (POST /api/whatif/simulate)."""

from __future__ import annotations

from fastapi import APIRouter

from api.models.trackside import WhatIfSimulateRequest, WhatIfSimulateResponse
from api.services.trackside_service import simulate_whatif_sweep

router = APIRouter(prefix="/api/whatif", tags=["whatif"])


@router.post("/simulate", response_model=WhatIfSimulateResponse)
def post_whatif_simulate(req: WhatIfSimulateRequest) -> WhatIfSimulateResponse:
    return simulate_whatif_sweep(req)
