"""Real-time 60 Hz WebSocket telemetry router (WS /ws/telemetry)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.core.config import WS_STREAM_HZ
from api.services.udp_bridge import telemetry_bridge

router = APIRouter(tags=["telemetry"])


@router.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    telemetry_bridge.register_ws(websocket)
    interval_s = 1.0 / max(WS_STREAM_HZ, 1.0)
    try:
        while True:
            frame = telemetry_bridge.get_next_frame()
            await websocket.send_json(frame.model_dump())
            await asyncio.sleep(interval_s)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        telemetry_bridge.unregister_ws(websocket)
