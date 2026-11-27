import React from "react";
import { WORKBENCH_ROUTES, WorkbenchRouteId } from "../../app/routes";

interface SidebarNavProps {
  activeRoute: WorkbenchRouteId;
  onSelectRoute: (route: WorkbenchRouteId) => void;
}

export function SidebarNav({ activeRoute, onSelectRoute }: SidebarNavProps) {
  return (
    <aside className="sidebar-nav" aria-label="Workbench navigation">
      <div
        style={{
          padding: "4px 8px 10px 8px",
          borderBottom: "1px solid var(--border-subtle)",
          marginBottom: "6px",
        }}
      >
        <div
          className="mono"
          style={{ fontSize: "11px", color: "var(--accent-gold)", letterSpacing: "0.08em" }}
        >
          TECNUN E-RACING // TER27
        </div>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
          ISO 8855 (+X fwd, +Y left, +Z up)
        </div>
      </div>

      {WORKBENCH_ROUTES.map((route) => {
        const isActive = route.id === activeRoute;
        return (
          <button
            key={route.id}
            type="button"
            className={`nav-item ${isActive ? "active" : ""}`}
            onClick={() => onSelectRoute(route.id)}
            title={route.description}
          >
            <span>{route.label}</span>
            <span className="mono" style={{ fontSize: "10px", opacity: 0.75 }}>
              {route.shortCode}
            </span>
          </button>
        );
      })}
    </aside>
  );
}
