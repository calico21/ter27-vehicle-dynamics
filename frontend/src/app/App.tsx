import React, { useState } from "react";
import { AppProviders } from "./providers";
import { WorkbenchRouteId } from "./routes";
import { AppShell } from "../components/layout/AppShell";
import { LivePitwallView } from "../components/telemetry/LivePitwallView";
import { SessionABComparator } from "../components/forensics/SessionABComparator";
import { Vehicle3DViewport } from "../components/digital_twin/Vehicle3DViewport";
import { TireModelExplorer } from "../components/tire_lab/TireModelExplorer";
import { TracksideSetupMatrix } from "../components/setup_matrix/TracksideSetupMatrix";
import { ValidationVaultBrowser } from "../components/validation/ValidationVaultBrowser";

function WorkbenchRouter({ activeRoute }: { activeRoute: WorkbenchRouteId }) {
  switch (activeRoute) {
    case "live_pitwall":
      return <LivePitwallView />;
    case "forensics":
      return <SessionABComparator />;
    case "digital_twin":
      return <Vehicle3DViewport />;
    case "tire_lab":
      return <TireModelExplorer />;
    case "setup_matrix":
      return <TracksideSetupMatrix />;
    case "validation_vault":
      return <ValidationVaultBrowser />;
    default:
      return <LivePitwallView />;
  }
}

export function App() {
  const [activeRoute, setActiveRoute] = useState<WorkbenchRouteId>("live_pitwall");

  return (
    <AppProviders>
      <AppShell activeRoute={activeRoute} onSelectRoute={setActiveRoute}>
        <WorkbenchRouter activeRoute={activeRoute} />
      </AppShell>
    </AppProviders>
  );
}
