import React from "react";
import { WorkbenchRouteId } from "../../app/routes";
import { SidebarNav } from "./SidebarNav";
import { TopStatusBar } from "./TopStatusBar";

interface AppShellProps {
  activeRoute: WorkbenchRouteId;
  onSelectRoute: (route: WorkbenchRouteId) => void;
  children: React.ReactNode;
}

export function AppShell({ activeRoute, onSelectRoute, children }: AppShellProps) {
  return (
    <div className="app-shell">
      <TopStatusBar />
      <SidebarNav activeRoute={activeRoute} onSelectRoute={onSelectRoute} />
      <main className="main-viewport">{children}</main>
    </div>
  );
}
