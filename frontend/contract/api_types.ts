/**
 * TeR27 Vehicle Dynamics & Telemetry Workbench -- TypeScript API Contract
 *
 * Source of truth: api/models/ (Pydantic v2) and frontend/contract/openapi.json.
 * Coordinate convention: ISO 8855 (+X forward, +Y left, +Z up).
 */

// ---------------------------------------------------------------------------
// Literal Unions & Enums
// ---------------------------------------------------------------------------

/** Active telemetry stream origin for WS /ws/telemetry. */
export type TelemetrySource = "godot_udp" | "dil_session" | "synthetic_fallback";

/** Sensor realism mode toggled in MirenaSim. */
export type SensorProfileName = "SENSOR_IDEAL" | "SENSOR_TRACK_REALISM";

/** Category of stored session available for A/B analysis. */
export type SessionKindName = "dil_recording" | "casadi_reference" | "godot_parity";

/** Hoosier 16.0x7.5-10 tire compound identifier. */
export type TireCompoundName = "R20" | "R25B";

/** Standardized track grip profile identifier. */
export type GripProfileName =
  | "COLD_TEST_OCT"
  | "FSG_RUBBERED"
  | "FSG_REAL_TRACK"
  | "TTC_LAB_PEAK";

/** Dynamic event supported by the CasADi simulation runner. */
export type SimulationEventName = "skidpad" | "accel" | "autocross" | "endurance";

/** Execution state of an asynchronous CasADi simulation job. */
export type SimulationJobStatusName = "queued" | "running" | "completed" | "failed";

/** Solver fidelity and driver envelope mode. */
export type SimulationModeName = "human_achievable" | "theoretical_optimal";

// ---------------------------------------------------------------------------
// Health & Real-Time Telemetry (GET /api/health, WS /ws/telemetry)
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: "ok";
  /** API semantic version. */
  version: string;
  /** True when live Godot UDP frames are being received. */
  udp_connected: boolean;
  /** Active telemetry source for WS /ws/telemetry. */
  telemetry_source: TelemetrySource;
  /** Number of connected WebSocket clients. */
  active_ws_clients: number;
  /** UDP port receiving MirenaSim telemetry (default 9876). */
  godot_udp_port: number;
  /** UDP port sending commands to MirenaSim (default 9877). */
  godot_cmd_port: number;
  /** Standardized lambda_muy grip targets (COLD_TEST_OCT=0.47, FSG_RUBBERED=0.57). */
  grip_profiles: Record<string, number>;
}

export interface VehicleState14Dof {
  /** Longitudinal velocity in ISO 8855 (m/s). */
  vx: number;
  /** Lateral velocity in ISO 8855 (m/s). */
  vy: number;
  /** Vertical velocity in ISO 8855 (m/s). */
  vz: number;
  /** Longitudinal acceleration (g). */
  ax_g: number;
  /** Lateral acceleration (g). */
  ay_g: number;
  /** Yaw rate omega_z (rad/s). */
  yaw_rate_rad_s: number;
  /** Vehicle sideslip angle beta (deg). */
  beta_deg: number;
  /** Front road-wheel steer angle (deg). */
  steer_deg: number;
  /** Normalized throttle command in [0, 1]. */
  throttle: number;
  /** Normalized brake command in [0, 1]. */
  brake: number;
  /** Global X position in ISO 8855 (m). */
  x_m: number;
  /** Global Y position in ISO 8855 (m). */
  y_m: number;
  /** Global Z position in ISO 8855 (m). */
  z_m: number;
  /** Global yaw heading (rad). */
  heading_rad: number;
}

export interface WheelTelemetry {
  /** Longitudinal tire force (N). */
  fx_n: number;
  /** Lateral tire force (N). */
  fy_n: number;
  /** Normal contact load (N). */
  fz_n: number;
  /** Tire slip angle alpha (deg). */
  slip_angle_deg: number;
  /** Longitudinal slip ratio kappa. */
  slip_ratio: number;
  /** Combined friction circle utilization (>= 0). */
  tire_util: number;
  /** Effective inclination angle (deg). */
  camber_deg: number;
  /** Suspension compression travel (m). */
  wheel_travel_m: number;
  /** Net wheel motor torque (Nm). */
  motor_torque_nm: number;
  /** Tire surface temperature (C). */
  temp_c: number;
}

export interface WheelsTelemetry {
  fl: WheelTelemetry;
  fr: WheelTelemetry;
  rl: WheelTelemetry;
  rr: WheelTelemetry;
}

export interface HvPowerTelemetry {
  /** HV DC tractive discharge power (kW). */
  p_dc_kw: number;
  /** HV DC regenerative charge power (kW). */
  p_regen_kw: number;
  /** Battery state of charge in [0, 100] (%). */
  soc_pct: number;
  /** FSAE EV 2.3 BSPD plausibility flag. */
  bspd_active: boolean;
  /** Torque vectoring yaw moment command (Nm). */
  tv_mz_nm: number;
}

export interface EkfStateTelemetry {
  /** EKF estimated longitudinal velocity (m/s). */
  ekf_vx: number;
  /** EKF estimated lateral velocity (m/s). */
  ekf_vy: number;
  /** EKF estimated yaw rate (rad/s). */
  ekf_yaw_rate: number;
  /** EKF estimated sideslip angle (deg). */
  ekf_beta_deg: number;
  /** CAN transport latency (ms). */
  sensor_latency_ms: number;
}

export interface DilDeltaTelemetry {
  /** Live time delta vs CasADi reference (s). */
  delta_time_s: number;
  /** Speed delta vs CasADi reference (m/s). */
  delta_vx_mps: number;
  /** Lateral accel delta vs CasADi reference (g). */
  delta_ay_g: number;
}

export interface TelemetryFrame {
  /** Simulation or stream timestamp (s). */
  timestamp_s: number;
  /** Telemetry origin. */
  source: TelemetrySource;
  /** Active vehicle identifier (e.g. ter27_4wd). */
  vehicle_id: string;
  /** Active track grip profile. */
  grip_profile: string;
  /** Active sensor realism profile. */
  sensor_profile: SensorProfileName;
  state: VehicleState14Dof;
  wheels: WheelsTelemetry;
  hv: HvPowerTelemetry;
  ekf: EkfStateTelemetry;
  dil_delta: DilDeltaTelemetry;
}

// ---------------------------------------------------------------------------
// Sessions & A/B Comparison (GET /api/sessions, GET /api/sessions/{session_id})
// ---------------------------------------------------------------------------

export interface SessionSummary {
  session_id: string;
  title: string;
  kind: SessionKindName;
  track_or_event: string;
  grip_profile: string;
  n_samples: number;
  duration_s: number;
  source_path: string;
}

export interface SessionsListResponse {
  sessions: SessionSummary[];
  total_count: number;
}

export interface TrajectorySample {
  t_s: number;
  s_m: number;
  x_m: number;
  y_m: number;
  vx_mps: number;
  ay_g: number;
  ax_g: number;
  beta_deg: number;
  steer_deg: number;
  mz_tv_nm: number;
  delta_t_s: number;
}

export interface SessionABComparisonResponse {
  session: SessionSummary;
  driver_trace: TrajectorySample[];
  casadi_reference_trace: TrajectorySample[];
  kpis: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Vehicle Overview (GET /api/vehicle/overview)
// ---------------------------------------------------------------------------

export interface GripProfileSpec {
  name: string;
  env_mu_scale: number;
  env_aero_cl_scale: number;
  lambda_muy_target?: number | null;
  track_temp_c?: number | null;
  tire_temp_c?: number | null;
}

export interface CornerHardpoints3D {
  lca_front: number[];
  lca_rear: number[];
  lbj: number[];
  uca_front: number[];
  uca_rear: number[];
  ubj: number[];
  pushrod_wishbone: number[];
  pushrod_rocker: number[];
  tie_outer: number[];
  tie_inner: number[];
  wheel_center: number[];
  contact_patch: number[];
  damper_chassis: number[];
  damper_rocker: number[];
  rocker_axis_p1: number[];
  rocker_pivot: number[];
  arb_link_rocker: number[];
  arb_arm_tip: number[];
  arb_chassis_piv: number[];
  spring_stiffness_n_m: number;
}

export interface SuspensionOverview {
  use_3d_kinematics: boolean;
  frame: string;
  spring_stiffness_front_n_m: number;
  spring_stiffness_rear_n_m: number;
  motion_ratio_front: number;
  motion_ratio_rear: number;
  arb_stiffness_front_nm_rad: number;
  arb_stiffness_rear_nm_rad: number;
  lltd_elastic_front: number;
  static_camber_front_deg: number;
  static_camber_rear_deg: number;
  caster_deg: number;
  kpi_deg: number;
  anti_dive_front: number;
  anti_squat_rear: number;
  front_left_hardpoints: CornerHardpoints3D;
  rear_left_hardpoints: CornerHardpoints3D;
}

export interface VehiclePhysicalParams {
  mass_kg: number;
  sprung_mass_kg: number;
  unsprung_mass_total_kg: number;
  wheelbase_m: number;
  track_front_m: number;
  track_rear_m: number;
  cg_height_m: number;
  front_weight_dist: number;
  ixx_kgm2: number;
  iyy_kgm2: number;
  izz_total_kgm2: number;
  cl_a_m2: number;
  cd_a_m2: number;
  aero_balance_front: number;
  wheel_radius_m: number;
  tire_compound: string;
  p_dc_max_kw: number;
  p_dc_regen_max_kw: number;
  motor_peak_torque_nm: number;
  gear_ratio: number;
  battery_capacity_kwh: number;
  battery_nominal_v: number;
}

export interface ValidationStatusSummary {
  vault_suites: number;
  total_figures: number;
  cold_test_oct_lambda_muy: number;
  fsg_rubbered_lambda_muy: number;
  skidpad_reference_time_s?: number | null;
  autocross_reference_time_s?: number | null;
  endurance_22km_time_s?: number | null;
  status: string;
}

export interface VehicleOverviewResponse {
  vehicle_id: string;
  display_name: string;
  drive_type: string;
  params: VehiclePhysicalParams;
  suspension: SuspensionOverview;
  grip_profiles: GripProfileSpec[];
  setup_bounds: Record<string, number[]>;
  validation_status: ValidationStatusSummary;
}

// ---------------------------------------------------------------------------
// Parametric Tire Evaluation (POST /api/tires/evaluate)
// ---------------------------------------------------------------------------

export interface TireEvaluateRequest {
  /** Hoosier 16.0x7.5-10 compound (default "R25B"). */
  compound?: TireCompoundName;
  /** Track grip profile (default "FSG_RUBBERED"). */
  grip_profile?: GripProfileName;
  /** Normal load Fz in [50, 3500] (N, default 750.0). */
  fz_n?: number;
  /** Camber angle in [-6, 6] (deg, default -1.5). */
  camber_deg?: number;
  /** Tire surface temperature in [5, 120] (C, default 75.0). */
  temp_c?: number;
  /** Inflation pressure in [55, 130] (kPa, default 83.0). */
  pressure_kpa?: number;
  /** Sweep resolution in [15, 201] (default 61). */
  n_points?: number;
}

export interface TireCurvePoint {
  /** Sweep abscissa (alpha_deg or kappa). */
  x: number;
  /** Evaluated force or moment (N or Nm). */
  y: number;
}

export interface FrictionEllipsePoint {
  angle_deg: number;
  fx_n: number;
  fy_n: number;
}

export interface TireEvaluateResponse {
  compound: TireCompoundName;
  grip_profile: GripProfileName;
  fz_n: number;
  camber_deg: number;
  temp_c: number;
  pressure_kpa: number;
  lambda_mux_eff: number;
  lambda_muy_eff: number;
  thermal_mu_factor: number;
  peak_fx_n: number;
  peak_fy_n: number;
  mu_x_eff: number;
  mu_y_eff: number;
  fy_vs_alpha: TireCurvePoint[];
  fx_vs_kappa: TireCurvePoint[];
  mz_vs_alpha: TireCurvePoint[];
  friction_ellipse: FrictionEllipsePoint[];
}

// ---------------------------------------------------------------------------
// Setups Matrix & Live Apply (GET /api/setups, POST /api/setups/apply)
// ---------------------------------------------------------------------------

export interface SetupValues {
  name: string;
  event: string;
  description: string;
  spring_rate_front_n_m: number;
  spring_rate_rear_n_m: number;
  arb_front_nm_rad: number;
  arb_rear_nm_rad: number;
  lltd: number;
  camber_front_deg: number;
  camber_rear_deg: number;
  toe_front_deg: number;
  toe_rear_deg: number;
  ride_height_front_m: number;
  ride_height_rear_m: number;
  rake_mm: number;
  brake_bias: number;
  tire_pressure_front_bar: number;
  tire_pressure_rear_bar: number;
  tv_map: string;
  power_limit_kw: number;
  control_law_mode: string;
}

export interface SetupPresetDiff {
  preset: SetupValues;
  deltas_vs_base: Record<string, number>;
  casadi_metrics: Record<string, unknown>;
}

export interface EventOptimizationGain {
  event: string;
  event_name: string;
  metric_label: string;
  base_time_s: number;
  optimal_time_s: number;
  delta_time_s: number;
  delta_pct: number;
  preset_file: string;
}

export interface SetupsMatrixResponse {
  base_setup: SetupValues;
  presets: SetupPresetDiff[];
  optimization_gains: EventOptimizationGain[];
  fs_points_base: Record<string, number>;
  fs_points_optimal: Record<string, number>;
  delta_total_pts: number;
}

export interface ApplySetupRequest {
  /** Preset name, e.g. "optimal_skidpad". */
  preset?: string | null;
  /** Optional full setup dictionary override. */
  setup?: Record<string, unknown> | null;
  /** Optional sensor profile switch. */
  sensor_profile?: SensorProfileName | null;
}

export interface ApplySetupResponse {
  acknowledged: boolean;
  udp_dispatched: boolean;
  mirena_sim_online: boolean;
  target_host: string;
  target_port: number;
  command_payload: Record<string, unknown>;
  message: string;
}

// ---------------------------------------------------------------------------
// CasADi Simulation Runner (POST /api/simulation/run, GET /api/simulation/status/{job_id})
// ---------------------------------------------------------------------------

export interface SimulationRunRequest {
  /** Dynamic event to solve (default "skidpad"). */
  event?: SimulationEventName;
  /** Grip profile (default "FSG_RUBBERED"). */
  grip_profile?: GripProfileName;
  /** Tire compound (default "R25B"). */
  compound?: TireCompoundName;
  /** Enable 7-parameter trackside setup co-optimization (default false). */
  optimize_setup?: boolean;
  /** Solver fidelity and driver envelope mode (default "human_achievable"). */
  mode?: SimulationModeName;
  /** Discretization intervals for full-lap OCP in [40, 500] (default 120). */
  n_intervals?: number;
  /** IPOPT maximum iterations in [50, 1200] (default 300). */
  max_iter?: number;
  /** Return cached validated CasADi reference metrics without spawning IPOPT (default false). */
  dry_run?: boolean;
}

export interface SimulationJobStatusResponse {
  job_id: string;
  event: SimulationEventName;
  grip_profile: string;
  compound: string;
  optimize_setup: boolean;
  status: SimulationJobStatusName;
  /** Completion percentage in [0, 100]. */
  progress_pct: number;
  created_at_utc: string;
  started_at_utc?: string | null;
  finished_at_utc?: string | null;
  command: string[];
  result?: Record<string, unknown> | null;
  error?: string | null;
  log_tail: string[];
}

// ---------------------------------------------------------------------------
// Validation Vault Catalog (GET /api/vault/catalog)
// ---------------------------------------------------------------------------

export interface VaultKpi {
  label: string;
  value: string;
  status: string;
}

export interface VaultFigure {
  filename: string;
  rel_path: string;
  title: string;
  size_kb: number;
  width_px: number;
  height_px: number;
}

export interface VaultSuite {
  code: string;
  folder: string;
  title: string;
  subtitle: string;
  tag: string;
  primary_metrics_json?: string | null;
  json_files: string[];
  csv_files: string[];
  n_figures: number;
  kpis: VaultKpi[];
  figures: VaultFigure[];
}

export interface VaultCatalogResponse {
  generated_utc: string;
  standard: string;
  vault_root: string;
  n_suites: number;
  total_pngs: number;
  total_jsons: number;
  total_png_size_mb: number;
  suites: VaultSuite[];
}
