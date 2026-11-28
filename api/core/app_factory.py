"""FastAPI application factory wiring middleware, lifecycle, and domain routers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import __version__
from api.core.config import settings
from api.core.lifecycle import lifespan
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


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.title,
        version=__version__,
        description=settings.description,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(telemetry_ws.router)
    app.include_router(vehicle.router)
    app.include_router(models_catalog.router)
    app.include_router(sessions.router)
    app.include_router(tires.router)
    app.include_router(setups.router)
    app.include_router(simulation.router)
    app.include_router(vault.router)
    app.include_router(trackside.router)
    app.include_router(whatif.router)

    return app
