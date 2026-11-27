import type {
  ApplySetupRequest,
  ApplySetupResponse,
  HealthResponse,
  SessionABComparisonResponse,
  SessionsListResponse,
  SetupsMatrixResponse,
  SimulationJobStatusResponse,
  SimulationRunRequest,
  TireEvaluateRequest,
  TireEvaluateResponse,
  VaultCatalogResponse,
  VehicleOverviewResponse,
} from "../types";

import mockDilComparison from "../../mock_data/dil_session_comparison.json";
import mockTireCurves from "../../mock_data/tire_evaluation_curves.json";
import mockSetupsMatrix from "../../mock_data/trackside_setups_matrix.json";
import mockVaultCatalog from "../../mock_data/validation_vault_catalog.json";

const DEFAULT_API_BASE = "http://localhost:8000";

export function getApiBaseUrl(): string {
  const envBase =
    (import.meta as unknown as { env?: Record<string, string | undefined> }).env
      ?.VITE_TER27_API_URL ?? DEFAULT_API_BASE;
  return envBase.replace(/\/$/, "");
}

export interface ApiResult<T> {
  data: T;
  live: boolean;
}

async function fetchJsonWithFallback<T>(
  path: string,
  fallback: T,
  init?: RequestInit
): Promise<ApiResult<T>> {
  try {
    const res = await fetch(`${getApiBaseUrl()}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const payload = (await res.json()) as T;
    return { data: payload, live: true };
  } catch {
    return { data: fallback, live: false };
  }
}

export function fetchHealth(): Promise<ApiResult<HealthResponse>> {
  const fallback: HealthResponse = {
    status: "ok",
    version: "1.0.0",
    udp_connected: false,
    telemetry_source: "synthetic_fallback",
    active_ws_clients: 0,
    godot_udp_port: 9876,
    godot_cmd_port: 9877,
    grip_profiles: { COLD_TEST_OCT: 0.47, FSG_RUBBERED: 0.57 },
  };
  return fetchJsonWithFallback<HealthResponse>("/api/health", fallback);
}

export function fetchVehicleOverview(): Promise<ApiResult<VehicleOverviewResponse>> {
  const emptyCorner = {
    lca_front: [0, 0, 0],
    lca_rear: [0, 0, 0],
    lbj: [0, 0, 0],
    uca_front: [0, 0, 0],
    uca_rear: [0, 0, 0],
    ubj: [0, 0, 0],
    pushrod_wishbone: [0, 0, 0],
    pushrod_rocker: [0, 0, 0],
    tie_outer: [0, 0, 0],
    tie_inner: [0, 0, 0],
    wheel_center: [0, 0, 0],
    contact_patch: [0, 0, 0],
    damper_chassis: [0, 0, 0],
    damper_rocker: [0, 0, 0],
    rocker_axis_p1: [0, 0, 0],
    rocker_pivot: [0, 0, 0],
    arb_link_rocker: [0, 0, 0],
    arb_arm_tip: [0, 0, 0],
    arb_chassis_piv: [0, 0, 0],
    spring_stiffness_n_m: 45000,
  };
  const fallback: VehicleOverviewResponse = {
    vehicle_id: "ter27_4wd",
    display_name: "TeR27 4WD Formula Student EV",
    drive_type: "4WD",
    params: {
      mass_kg: 263.0,
      sprung_mass_kg: 223.0,
      unsprung_mass_total_kg: 40.0,
      wheelbase_m: 1.54,
      track_front_m: 1.24,
      track_rear_m: 1.22,
      cg_height_m: 0.285,
      front_weight_dist: 0.48,
      ixx_kgm2: 32.0,
      iyy_kgm2: 95.0,
      izz_total_kgm2: 112.0,
      cl_a_m2: 2.35,
      cd_a_m2: 1.05,
      aero_balance_front: 0.45,
      wheel_radius_m: 0.203,
      tire_compound: "R25B",
      p_dc_max_kw: 80.0,
      p_dc_regen_max_kw: 25.0,
      motor_peak_torque_nm: 21.0,
      gear_ratio: 13.5,
      battery_capacity_kwh: 6.8,
      battery_nominal_v: 540.0,
    },
    suspension: {
      use_3d_kinematics: true,
      frame: "ISO_8855",
      spring_stiffness_front_n_m: 45000,
      spring_stiffness_rear_n_m: 50000,
      motion_ratio_front: 0.92,
      motion_ratio_rear: 0.92,
      arb_stiffness_front_nm_rad: 320,
      arb_stiffness_rear_nm_rad: 280,
      lltd_elastic_front: 0.5,
      static_camber_front_deg: -1.5,
      static_camber_rear_deg: -1.2,
      caster_deg: 5.0,
      kpi_deg: 6.5,
      anti_dive_front: 0.15,
      anti_squat_rear: 0.12,
      front_left_hardpoints: emptyCorner,
      rear_left_hardpoints: emptyCorner,
    },
    grip_profiles: [
      {
        name: "COLD_TEST_OCT",
        env_mu_scale: 0.47,
        env_aero_cl_scale: 1.0,
        lambda_muy_target: 0.47,
        track_temp_c: 18,
        tire_temp_c: 35,
      },
      {
        name: "FSG_RUBBERED",
        env_mu_scale: 0.57,
        env_aero_cl_scale: 1.0,
        lambda_muy_target: 0.57,
        track_temp_c: 32,
        tire_temp_c: 75,
      },
    ],
    setup_bounds: {
      lltd: [0.35, 0.62],
      camber_f_deg: [-2.5, -0.2],
      camber_r_deg: [-2.2, -0.2],
      toe_f_deg: [-0.5, 0.5],
      toe_r_deg: [0.0, 0.6],
      brake_bias: [0.55, 0.78],
      rake_mm: [-8.0, 8.0],
    },
    validation_status: {
      vault_suites: 10,
      total_figures: 67,
      cold_test_oct_lambda_muy: 0.47,
      fsg_rubbered_lambda_muy: 0.57,
      status: "OFFLINE_FALLBACK",
    },
  };
  return fetchJsonWithFallback<VehicleOverviewResponse>("/api/vehicle/overview", fallback);
}

export function fetchSessionsList(): Promise<ApiResult<SessionsListResponse>> {
  const sample = mockDilComparison as unknown as SessionABComparisonResponse;
  const fallback: SessionsListResponse = {
    sessions: sample.session ? [sample.session] : [],
    total_count: sample.session ? 1 : 0,
  };
  return fetchJsonWithFallback<SessionsListResponse>("/api/sessions", fallback);
}

export function fetchSessionComparison(
  sessionId: string
): Promise<ApiResult<SessionABComparisonResponse>> {
  const fallback = mockDilComparison as unknown as SessionABComparisonResponse;
  return fetchJsonWithFallback<SessionABComparisonResponse>(
    `/api/sessions/${encodeURIComponent(sessionId)}`,
    fallback
  );
}

export function evaluateTireModel(
  req: TireEvaluateRequest
): Promise<ApiResult<TireEvaluateResponse>> {
  const fallback = mockTireCurves as unknown as TireEvaluateResponse;
  return fetchJsonWithFallback<TireEvaluateResponse>("/api/tires/evaluate", fallback, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function fetchSetupsMatrix(): Promise<ApiResult<SetupsMatrixResponse>> {
  const fallback = mockSetupsMatrix as unknown as SetupsMatrixResponse;
  return fetchJsonWithFallback<SetupsMatrixResponse>("/api/setups", fallback);
}

export function applySetupPreset(
  req: ApplySetupRequest
): Promise<ApiResult<ApplySetupResponse>> {
  const fallback: ApplySetupResponse = {
    acknowledged: false,
    udp_dispatched: false,
    mirena_sim_online: false,
    target_host: "127.0.0.1",
    target_port: 9877,
    command_payload: req as unknown as Record<string, unknown>,
    message: "Backend offline: command queued in mock mode.",
  };
  return fetchJsonWithFallback<ApplySetupResponse>("/api/setups/apply", fallback, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function launchSimulationJob(
  req: SimulationRunRequest
): Promise<ApiResult<SimulationJobStatusResponse>> {
  const fallback: SimulationJobStatusResponse = {
    job_id: "offline-dry-run",
    event: req.event ?? "skidpad",
    grip_profile: req.grip_profile ?? "FSG_RUBBERED",
    compound: req.compound ?? "R25B",
    optimize_setup: req.optimize_setup ?? false,
    status: "completed",
    progress_pct: 100,
    created_at_utc: new Date().toISOString(),
    started_at_utc: new Date().toISOString(),
    finished_at_utc: new Date().toISOString(),
    command: ["offline-mock"],
    result: { mode: "offline_mock" },
    error: null,
    log_tail: ["Backend offline: showing mock simulation completion."],
  };
  return fetchJsonWithFallback<SimulationJobStatusResponse>("/api/simulation/run", fallback, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function pollSimulationStatus(
  jobId: string
): Promise<ApiResult<SimulationJobStatusResponse>> {
  const fallback: SimulationJobStatusResponse = {
    job_id: jobId,
    event: "skidpad",
    grip_profile: "FSG_RUBBERED",
    compound: "R25B",
    optimize_setup: false,
    status: "completed",
    progress_pct: 100,
    created_at_utc: new Date().toISOString(),
    started_at_utc: new Date().toISOString(),
    finished_at_utc: new Date().toISOString(),
    command: ["offline-mock"],
    result: { mode: "offline_mock" },
    error: null,
    log_tail: ["Backend offline: mock job status."],
  };
  return fetchJsonWithFallback<SimulationJobStatusResponse>(
    `/api/simulation/status/${encodeURIComponent(jobId)}`,
    fallback
  );
}

export function fetchVaultCatalog(): Promise<ApiResult<VaultCatalogResponse>> {
  const fallback = mockVaultCatalog as unknown as VaultCatalogResponse;
  return fetchJsonWithFallback<VaultCatalogResponse>("/api/vault/catalog", fallback);
}
