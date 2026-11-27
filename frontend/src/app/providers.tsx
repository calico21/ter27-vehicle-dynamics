import React, { useEffect } from "react";
import { TelemetryProvider, useTelemetryStore } from "../state/telemetryStore";

function TelemetryStreamBootstrap({ children }: { children: React.ReactNode }) {
  const { startStream } = useTelemetryStore();

  useEffect(() => {
    const stop = startStream();
    return () => {
      stop();
    };
  }, [startStream]);

  return <>{children}</>;
}

export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <TelemetryProvider>
      <TelemetryStreamBootstrap>{children}</TelemetryStreamBootstrap>
    </TelemetryProvider>
  );
}
