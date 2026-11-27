import React, { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { TelemetryFrame } from "../types";
import {
  connectTelemetrySocket,
  getOfflineMockTelemetryFrame,
  TelemetryConnectionStatus,
} from "../services/telemetrySocket";

const MAX_HISTORY_FRAMES = 300;

interface TelemetryStoreValue {
  frame: TelemetryFrame;
  history: TelemetryFrame[];
  connectionStatus: TelemetryConnectionStatus;
  startStream: () => () => void;
}

const TelemetryContext = createContext<TelemetryStoreValue | null>(null);

export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  const [frame, setFrame] = useState<TelemetryFrame>(() => getOfflineMockTelemetryFrame());
  const [history, setHistory] = useState<TelemetryFrame[]>(() => [getOfflineMockTelemetryFrame()]);
  const [connectionStatus, setConnectionStatus] =
    useState<TelemetryConnectionStatus>("offline_mock");

  const startStream = useCallback(() => {
    return connectTelemetrySocket(
      (nextFrame) => {
        setFrame(nextFrame);
        setHistory((prev) => {
          const updated = [...prev, nextFrame];
          return updated.length > MAX_HISTORY_FRAMES
            ? updated.slice(updated.length - MAX_HISTORY_FRAMES)
            : updated;
        });
      },
      (status) => {
        setConnectionStatus(status);
      }
    );
  }, []);

  const value = useMemo<TelemetryStoreValue>(
    () => ({ frame, history, connectionStatus, startStream }),
    [frame, history, connectionStatus, startStream]
  );

  return React.createElement(TelemetryContext.Provider, { value }, children);
}

export function useTelemetryStore(): TelemetryStoreValue {
  const ctx = useContext(TelemetryContext);
  if (!ctx) {
    throw new Error("useTelemetryStore must be used inside TelemetryProvider");
  }
  return ctx;
}
