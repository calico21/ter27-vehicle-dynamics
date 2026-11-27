export type WorkbenchRouteId =
  | "live_pitwall"
  | "forensics"
  | "digital_twin"
  | "tire_lab"
  | "setup_matrix"
  | "validation_vault";

export interface WorkbenchRouteSpec {
  id: WorkbenchRouteId;
  label: string;
  shortCode: string;
  description: string;
}

export const WORKBENCH_ROUTES: WorkbenchRouteSpec[] = [
  {
    id: "live_pitwall",
    label: "Live Pit-Wall",
    shortCode: "WS 60Hz",
    description: "Real-time 14-DOF telemetry stream, 4-corner tire utilization, HV power, and EKF status",
  },
  {
    id: "forensics",
    label: "Session A/B Forensics",
    shortCode: "DiL vs OPT",
    description: "Driver-in-the-Loop vs CasADi optimal reference trace comparison and sector deltas",
  },
  {
    id: "digital_twin",
    label: "3D Digital Twin",
    shortCode: "14-DOF",
    description: "TeR27 4WD double-wishbone kinematics, wheel travel, and ISO 8855 hardpoints",
  },
  {
    id: "tire_lab",
    label: "MF 6.2 Tire Lab",
    shortCode: "R20/R25B",
    description: "Parametric Hoosier 16.0x7.5-10 lateral, longitudinal, self-aligning, and friction ellipse sweep",
  },
  {
    id: "setup_matrix",
    label: "Setup Matrix & LTS",
    shortCode: "SETUP",
    description: "Trackside setup differential matrix, live UDP push to MirenaSim, and CasADi simulation launcher",
  },
  {
    id: "validation_vault",
    label: "Validation Vault",
    shortCode: "01..10",
    description: "10-suite verification catalog with KPI pills and 300-DPI figure browser",
  },
];
