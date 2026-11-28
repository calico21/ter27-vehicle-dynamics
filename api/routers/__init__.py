"""Domain API routers for the TeR27 Vehicle Dynamics & Telemetry API."""

from api.routers import (
    health,
    models_catalog,
    sessions,
    setups,
    simulation,
    telemetry_ws,
    tires,
    trackside,
    vault,
    vehicle,
    whatif,
)

__all__ = [
    "health",
    "models_catalog",
    "sessions",
    "setups",
    "simulation",
    "telemetry_ws",
    "tires",
    "trackside",
    "vault",
    "vehicle",
    "whatif",
]
