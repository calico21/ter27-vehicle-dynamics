import type { TelemetryFrame } from "../types";
import mockLiveFrame from "../../mock_data/live_telemetry_frame.json";

export type TelemetryConnectionStatus = "connected" | "reconnecting" | "offline_mock";

const DEFAULT_API_BASE = "http://localhost:8000";

function getWsUrl(): string {
  const envBase =
    (import.meta as unknown as { env?: Record<string, string | undefined> }).env
      ?.VITE_TER27_API_URL ?? DEFAULT_API_BASE;
  return envBase.replace(/^http/, "ws").replace(/\/$/, "") + "/ws/telemetry";
}

export function getOfflineMockTelemetryFrame(): TelemetryFrame {
  return mockLiveFrame as unknown as TelemetryFrame;
}

export function connectTelemetrySocket(
  onFrame: (frame: TelemetryFrame) => void,
  onStatus: (status: TelemetryConnectionStatus) => void
): () => void {
  let ws: WebSocket | null = null;
  let retryMs = 500;
  let closed = false;
  let timerId: number | null = null;

  const open = () => {
    if (closed) return;
    try {
      ws = new WebSocket(getWsUrl());
    } catch {
      onStatus("offline_mock");
      onFrame(getOfflineMockTelemetryFrame());
      return;
    }

    ws.onopen = () => {
      retryMs = 500;
      onStatus("connected");
    };

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as TelemetryFrame;
        onFrame(parsed);
      } catch {
        // Ignore malformed telemetry frame
      }
    };

    ws.onerror = () => {
      ws?.close();
    };

    ws.onclose = () => {
      if (closed) return;
      onStatus("reconnecting");
      onFrame(getOfflineMockTelemetryFrame());
      timerId = window.setTimeout(() => {
        retryMs = Math.min(retryMs * 2, 8000);
        open();
      }, retryMs);
    };
  };

  open();

  return () => {
    closed = true;
    if (timerId !== null) {
      window.clearTimeout(timerId);
    }
    ws?.close();
  };
}
