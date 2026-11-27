"""Domain API routers for the TeR27 Vehicle Dynamics & Telemetry API."""

from api.routers import (
    health,
    sessions,
    setups,
    simulation,
    telemetry_ws,
    tires,
    vault,
    vehicle,
)

__all__ = [
    "health",
    "sessions",
    "setups",
    "simulation",
    "telemetry_ws",
    "tires",
    "vault",
    "vehicle",
]
