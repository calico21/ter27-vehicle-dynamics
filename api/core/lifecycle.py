"""FastAPI lifespan hooks managing the UDP telemetry bridge."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.services.udp_bridge import telemetry_bridge


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await telemetry_bridge.start()
    try:
        yield
    finally:
        await telemetry_bridge.stop()
