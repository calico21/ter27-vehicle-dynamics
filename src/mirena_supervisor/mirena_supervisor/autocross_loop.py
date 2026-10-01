"""Autocross PID sweep -- runs the full sim + supervisor + autocross pipeline
once per set of lateral PID gains and reports lap time and cones hit per set.

The sweep is driven by a CSV file (default: pid_sweep_results.csv). Each row
holds one parameter set; the script reads the CSV, skips rows that already
have a lap_time_s result, and runs the next untested combination. Results are
written back to the CSV after every run, so the sweep is fully resumable:
launch it again whenever and it picks up exactly where it left off.

For every untested parameter set in the CSV:
  1. Launch the sim and the supervisor, forwarding the set to the mission
      as launch arguments (the supervisor passes them to autocross.launch.py)
  2. Wait for the system to accept AS_status, then select Autocross
  3. Wait for the mission pipeline to report READY
  4. Release the car (AS_STATUS_DRIVING)
  5. Wait for /lap_status or a timeout, then tear down and restart

Usage:
    ros2 run mirena_supervisor autocross_loop.py
    ros2 run mirena_supervisor autocross_loop.py --ros-args -p csv_path:=/path/to/sweep.csv
"""

import csv
import os
import signal
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from mirena_common.msg import AsInfo, MissionInfo, LapStatus


# How long to wait for a lap before giving up on this run.
LAP_TIMEOUT_SEC: float = 60*3

# How long to wait for the sim + supervisor to come up before sending AS_STATUS_READY.
STARTUP_WAIT_SEC: float = 10.0

# How long to wait for the mission pipeline to report READY.
MISSION_READY_TIMEOUT_SEC: float = 60.0

# Cooldown between runs, seconds.
COOLDOWN_SEC: float = 3.0

# Path to the sweep CSV. Can be overridden with the csv_path ROS parameter.
CSV_PATH: str = 'pid_sweep_results_2.csv'

# The lateral PID gains each run tries, in sweep order. Keys are the
# mirena_control_pid parameter names; autocross.launch.py declares them as
# launch arguments defaulting to the baseline below, so a mission started by
# hand runs exactly as it did before the sweep existed.
#
# The sweep is a full factorial grid: 5 x 5 x 5 x 8 = 1000 combinations.
# Each parameter spans a plausible range around the baseline
# (kp 0.1-0.5 rad/m, ki 0-0.08, kd 0-0.2, tau_d 0.02-0.16 s); the baseline
# itself is on the grid.
def _linspace(start: float, stop: float, n: int) -> List[float]:
    """n evenly spaced values from start to stop, rounded for the launch args."""
    if n == 1:
        return [round(start, 4)]
    step = (stop - start) / (n - 1)
    return [round(start + i * step, 4) for i in range(n)]


PID_SWEEP: List[Dict[str, float]] = [
    {
        'cross_track_kp': kp,
        'cross_track_ki': ki,
        'cross_track_kd': kd,
        'cross_track_tau_d': tau_d,
    }
    for kp in _linspace(0.10, 0.50, 5)
    for ki in _linspace(0.02, 0.08, 5)
    for kd in _linspace(0.1, 0.20, 5)
    for tau_d in _linspace(0.02, 0.16, 8)
]
assert len(PID_SWEEP) == 1000


def generate_sweep_csv(path: str) -> None:
    """Create the sweep CSV from PID_SWEEP with all result columns empty."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cross_track_kp', 'cross_track_ki', 'cross_track_kd',
                         'cross_track_tau_d', 'lap_time_s', 'cones_hit'])
        for params in PID_SWEEP:
            writer.writerow([
                params['cross_track_kp'], params['cross_track_ki'],
                params['cross_track_kd'], params['cross_track_tau_d'],
                '', '',
            ])


def read_sweep_csv(path: str) -> List[Tuple[int, Dict[str, float], bool]]:
    """Read the sweep CSV.

    Returns a list of (row_index, params, has_result) tuples, where row_index
    is the 0-based data row index (excluding the header) and has_result is
    True when the lap_time_s column is non-empty.
    """
    rows: List[Tuple[int, Dict[str, float], bool]] = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            params = {
                'cross_track_kp': float(row['cross_track_kp']),
                'cross_track_ki': float(row['cross_track_ki']),
                'cross_track_kd': float(row['cross_track_kd']),
                'cross_track_tau_d': float(row['cross_track_tau_d']),
            }
            has_result = bool(row.get('lap_time_s', '').strip())
            rows.append((i, params, has_result))
    return rows


def update_csv_row(path: str, row_index: int,
                   lap_time: Optional[float], cones_hit: Optional[int]) -> None:
    """Write a test result back to its CSV row immediately."""
    with open(path, 'r', newline='') as f:
        rows = list(csv.reader(f))

    # row_index is 0-based data row; +1 skips the header
    data_row = rows[row_index + 1]
    data_row[4] = f'{lap_time:.3f}' if lap_time is not None else ''
    data_row[5] = str(cones_hit) if cones_hit is not None else ''

    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)


def claim_csv_row(path: str, row_index: int) -> None:
    """Claim a row by writing 0,0 -- signals to other parallel instances
    that this setup is already being tested."""
    update_csv_row(path, row_index, 0.0, 0)


class ConeHitError(Exception):
    """Raised when a cone is hit during a sweep run, to stop the entire sweep."""
    pass


class AutocrossLoop(Node):
    def __init__(self) -> None:
        super().__init__('autocross_loop')

        # QoS for /as_status: the sim publishes it, we publish it too.
        # Use RELIABLE + KEEP_LAST(10) to match the sim's publisher.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._as_pub = self.create_publisher(AsInfo, '/as_status', qos)
        self._mission_sub = self.create_subscription(
            MissionInfo, '/system/mission_status', self._mission_callback, 10)
        self._lap_sub = self.create_subscription(
            LapStatus, '/lap_status', self._lap_callback, 10)

        self._mission_ready = False
        self._lap_received = False
        self._lap_time: Optional[float] = None
        self._cones_hit: Optional[int] = None
        self._cone_hit_flag = False

    def _mission_callback(self, msg: MissionInfo) -> None:
        if msg.status == MissionInfo.MISSION_STATUS_READY:
            self._mission_ready = True

    def _lap_callback(self, msg: LapStatus) -> None:
        self._lap_received = True
        self._lap_time = msg.lap_time
        self._cones_hit = msg.cones_hit
        if msg.cones_hit > 0:
            self._cone_hit_flag = True
            self.get_logger().info(
                f"Cone hit! Total cones: {msg.cones_hit} | Time: {msg.lap_time:.3f}s")
        else:
            self.get_logger().info(
                f"Lap finished! Time: {msg.lap_time:.3f}s | Cones hit: {msg.cones_hit}")

    def reset(self) -> None:
        self._mission_ready = False
        self._lap_received = False
        self._lap_time = None
        self._cones_hit = None
        self._cone_hit_flag = False

    def send_as_status(self, status: int, mission: int = 0) -> None:
        msg = AsInfo()
        msg.status = status
        msg.mission = mission
        self._as_pub.publish(msg)

    def wait_for_mission_ready(self, timeout: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._mission_ready:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def wait_for_lap_or_timeout(self, timeout: float) -> bool:
        """Returns True if a lap was received or a cone was hit, False on timeout."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._cone_hit_flag:
                return True
            if self._lap_received:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False


def launch_stack(pid_params: Dict[str, float]) -> List[subprocess.Popen]:
    """Launch the sim and the supervisor with one PID set forwarded to the mission.

    The supervisor starts the mission launch file itself once the AS selects
    Autocross, forwarding its `mission_launch_arguments` parameter to it --
    which is how the gains below reach mirena_control_pid. This replicates
    sim.launch.py (same two nodes, same parameters) with the sweep's gains
    added to that argument list. Each node gets its own session so teardown
    can signal the whole process group.

    The sim, the supervisor and the mission nodes they spawn all inherit
    these processes' stdout/stderr, so every node's log output (including
    debug prints) is shown in the terminal; the sweep's own progress and
    the final results table are unaffected.
    """
    mission_args = ['use_sim_time:=true'] + [
        f'{name}:={value}' for name, value in pid_params.items()]

    sim = subprocess.Popen(
        ['ros2', 'run', 'mirena_sim', 'mirena_sim',
         '--ros-args',
         '-p', 'use_sim_time:=true',
         '-p', 'publish_sim_time:=true'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    supervisor = subprocess.Popen(
        ['ros2', 'run', 'mirena_supervisor', 'mirena_supervisor',
         '--ros-args',
         '-p', 'use_sim_time:=true',
         '-p', 'mission_launch_package:=mirena_supervisor',
         '-p', 'mission_launch_directory:=launch/missions',
         '-p', f'mission_launch_arguments:=[{", ".join(mission_args)}]'],
        start_new_session=True)
    return [sim, supervisor]


def child_pids(pid: Optional[int]) -> List[int]:
    """PIDs of the direct children of `pid` (empty if it has none or is gone)."""
    if pid is None:
        return []
    try:
        out = subprocess.run(
            ['ps', '--ppid', str(pid), '-o', 'pid='],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [int(line) for line in out.split()]


def signal_group(pid: Optional[int], sig: int) -> None:
    """Signal a process group, ignoring one that is already gone."""
    if pid is None:
        return
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def teardown(procs: List[subprocess.Popen]) -> None:
    """Stop the sim, the supervisor and the mission between runs.

    The supervisor spawns the mission as a child in its own process group, so
    it is collected before the supervisor dies and signalled afterwards --
    left alive it would keep running (and holding the car) for the rest of
    the sweep.
    """
    supervisor = procs[1]
    mission_pids = child_pids(supervisor.pid)

    for proc in procs:
        signal_group(proc.pid, signal.SIGTERM)
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            signal_group(proc.pid, signal.SIGKILL)
            proc.wait()

    for pid in mission_pids:
        signal_group(pid, signal.SIGTERM)
    time.sleep(1)
    for pid in mission_pids:
        signal_group(pid, signal.SIGKILL)


def run_once(loop: AutocrossLoop, pid_params: Dict[str, float],
             run_number: int, total: int
             ) -> Tuple[Optional[float], Optional[int]]:
    """One full iteration: launch, drive, wait for lap, teardown.

    Returns the lap time and cones hit, or (None, None) if the run produced
    no lap.
    """
    loop.get_logger().info(
        f"=== Run {run_number}/{total}: "
        + ", ".join(f"{k}={v}" for k, v in pid_params.items()) + " ===")
    loop.reset()

    procs = launch_stack(pid_params)

    try:
        # 1. Wait for the system to come up
        loop.get_logger().info(f"Waiting {STARTUP_WAIT_SEC:.0f}s for sim + supervisor...")
        time.sleep(STARTUP_WAIT_SEC)

        # 2. Select Autocross mission (AS_STATUS_READY)
        loop.get_logger().info("Sending AS_STATUS_READY + AUTOCROSS mission...")
        loop.send_as_status(AsInfo.AS_STATUS_READY, AsInfo.AS_MISSION_AUTOCROSS)

        # 3. Wait for the mission pipeline to be READY
        loop.get_logger().info("Waiting for mission READY...")
        if not loop.wait_for_mission_ready(MISSION_READY_TIMEOUT_SEC):
            loop.get_logger().error(
                f"Mission did not become READY within {MISSION_READY_TIMEOUT_SEC:.0f}s. "
                "Aborting run.")
            return None, None

        # 4. Release the car
        loop.get_logger().info("Mission READY. Sending AS_STATUS_DRIVING...")
        time.sleep(5)
        loop.send_as_status(AsInfo.AS_STATUS_DRIVING, AsInfo.AS_MISSION_AUTOCROSS)

        # 5. Wait for lap_status or timeout -- tear down early if a cone is hit
        loop.get_logger().info(
            f"Waiting for /lap_status (timeout: {LAP_TIMEOUT_SEC:.0f}s)...")
        got_lap = loop.wait_for_lap_or_timeout(LAP_TIMEOUT_SEC)

        if loop._cone_hit_flag:
            loop.get_logger().info(
                f"Cone hit detected ({loop._cones_hit} cones). Stopping sweep.")
        elif got_lap:
            loop.get_logger().info("Lap complete. Tearing down.")
        else:
            loop.get_logger().warn(
                f"No lap after {LAP_TIMEOUT_SEC:.0f}s. Tearing down.")

        return loop._lap_time, loop._cones_hit

    finally:
        # 6. Kill the sim, the supervisor and the mission
        loop.get_logger().info("Killing sim + supervisor...")
        teardown(procs)


def print_results(results: List[Tuple[Dict[str, float], Optional[float], Optional[int]]]) -> None:
    """The sweep's only summary: each parameter set's lap time and cones hit."""
    print()
    print('=' * 82)
    print('Autocross PID sweep results')
    print('=' * 82)
    header = (f"{'cross_track_kp':>14} {'cross_track_ki':>14} {'cross_track_kd':>14} "
              f"{'cross_track_tau_d':>17} | {'lap_time_s':>10} {'cones_hit':>9}")
    print(header)
    print('-' * len(header))
    for params, lap_time, cones_hit in results:
        t = f'{lap_time:.3f}' if lap_time is not None else '---'
        c = str(cones_hit) if cones_hit is not None else '---'
        print(f"{params['cross_track_kp']:>14.3f} {params['cross_track_ki']:>14.3f} "
              f"{params['cross_track_kd']:>14.3f} {params['cross_track_tau_d']:>17.3f} | "
              f"{t:>10} {c:>9}")
    print('=' * 82)


def main() -> None:
    rclpy.init()
    loop = AutocrossLoop()

    # Allow overriding the CSV path via a ROS parameter
    loop.declare_parameter('csv_path', CSV_PATH)
    csv_path = loop.get_parameter('csv_path').value

    # Generate the CSV from the grid if it doesn't exist yet
    if not os.path.exists(csv_path):
        loop.get_logger().info(f"CSV not found, generating {csv_path} from PID_SWEEP grid...")
        generate_sweep_csv(csv_path)

    # Read the sweep CSV and figure out what's left to do
    sweep_rows = read_sweep_csv(csv_path)
    total = len(sweep_rows)
    completed = sum(1 for _, _, has_result in sweep_rows if has_result)
    remaining = total - completed

    loop.get_logger().info(
        f"Sweep CSV: {csv_path} -- {total} total, {completed} already done, "
        f"{remaining} remaining")

    if remaining == 0:
        loop.get_logger().info("All runs already have results. Nothing to do.")
        loop.destroy_node()
        rclpy.shutdown()
        return

    run_count = 0
    try:
        for row_index, pid_params, has_result in sweep_rows:
            if has_result:
                continue

            # Claim this row so parallel instances skip it
            claim_csv_row(csv_path, row_index)
            loop.get_logger().info(
                f"Claimed row {row_index + 1}: "
                + ", ".join(f"{k}={v}" for k, v in pid_params.items()))

            run_count += 1
            lap_time, cones_hit = run_once(loop, pid_params, run_count, remaining)

            # Overwrite the claim with the actual result
            update_csv_row(csv_path, row_index, lap_time, cones_hit)

            if run_count < remaining:
                loop.get_logger().info(
                    f"Cooling down {COOLDOWN_SEC:.0f}s before next run...")
                time.sleep(COOLDOWN_SEC)
    except ConeHitError as e:
        loop.get_logger().info(str(e))
    except KeyboardInterrupt:
        loop.get_logger().info("Interrupted. Shutting down.")
    finally:
        loop.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # Re-read the CSV to print the full results table
    sweep_rows = read_sweep_csv(csv_path)
    results: List[Tuple[Dict[str, float], Optional[float], Optional[int]]] = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            params = {
                'cross_track_kp': float(row['cross_track_kp']),
                'cross_track_ki': float(row['cross_track_ki']),
                'cross_track_kd': float(row['cross_track_kd']),
                'cross_track_tau_d': float(row['cross_track_tau_d']),
            }
            lap_time = float(row['lap_time_s']) if row['lap_time_s'].strip() else None
            cones_hit = int(row['cones_hit']) if row['cones_hit'].strip() else None
            results.append((params, lap_time, cones_hit))

    print(f"Full results written to {csv_path}")


if __name__ == '__main__':
    main()
