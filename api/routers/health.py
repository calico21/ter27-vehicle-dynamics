"""Health and telemetry bridge diagnostics router (GET /api/health)."""

from __future__ import annotations

from fastapi import APIRouter

from api import __version__
from api.core.config import UDP_COMMAND_PORT, UDP_TELEMETRY_PORT
from api.models.telemetry import HealthResponse
from api.services.setup_service import get_grip_profile_targets
from api.services.udp_bridge import telemetry_bridge

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        udp_connected=telemetry_bridge.is_udp_connected,
        telemetry_source=telemetry_bridge.active_source,
        active_ws_clients=telemetry_bridge.active_ws_clients,
        godot_udp_port=UDP_TELEMETRY_PORT,
        godot_cmd_port=UDP_COMMAND_PORT,
        grip_profiles=get_grip_profile_targets(),
    )
