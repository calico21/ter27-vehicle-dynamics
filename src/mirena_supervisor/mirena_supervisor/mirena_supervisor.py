import os
import json
import math
from typing import Dict, Any, List, Optional, Set, Callable

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.client import Client
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
from rclpy.timer import Timer
from rclpy.task import Future
from rclpy.parameter import parameter_dict_from_yaml_file
from rclpy.parameter_client import AsyncParameterClient
from ament_index_python.packages import get_package_share_directory

from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition

# Adjust the import path if your package name differs
from mirena_common.msg import Heartbeat, AsInfo, MissionInfo

MAX_REPORT_SLACK_SEC: float = 3600  # 500ms threshold — node considered dead if exceeded



# Node names whose heartbeat reports feed lap counting.
SLAM_NODE_NAME: str = 'mirena_slam'
PLANNING_NODE_NAME: str = 'mirena_planning'
CONTROL_NODE_NAME: str = 'mirena_control'


mission_laps = {
    'TRACKDRIVE': 10,
    'AUTOCROSS': 1,
}

# AsInfo mission enum -> mission_config YAML basename
mission_names = {
    AsInfo.AS_MISSION_ACCELERATION: 'ACCELERATION',
    AsInfo.AS_MISSION_SKIDPAD: 'SKIDPAD',
    AsInfo.AS_MISSION_TRACKDRIVE: 'TRACKDRIVE',
    AsInfo.AS_MISSION_BRAKE_TEST: 'BRAKE_TEST',
    AsInfo.AS_MISSION_INSPECTION: 'INSPECTION',
    AsInfo.AS_MISSION_AUTOCROSS: 'AUTOCROSS',
}

class MirenaSupervisor(Node):
    def __init__(self) -> None:
        super().__init__('mirena_supervisor')
        self.get_logger().info("Mirena Supervisor Initialized")

        # --- Parameters ---
        self.declare_parameter('nodes_to_monitor', ['mirena_slam', 'mirena_planning', 'mirena_control'])

        self.nodes_to_monitor: List[str] = self.get_parameter('nodes_to_monitor').get_parameter_value().string_array_value

        # --- State Members (Stored as full ROS messages) ---
        self._as_info: AsInfo = AsInfo()
        self._as_info.status = AsInfo.AS_STATUS_OFF
        self._as_info.mission = AsInfo.AS_MISSION_NONE

        self._mission_info: MissionInfo = MissionInfo()
        self._mission_info.status = MissionInfo.MISSION_STATUS_INACTIVE

        # --- Tracking Dictionaries ---
        self._master_report: Dict[str, Dict[str, Any]] = {}
        self._node_heartbeats: Dict[str, Time] = {}
        self._lifecycle_clients: Dict[str, Client] = {}
        self._param_clients: Dict[str, AsyncParameterClient] = {}

        # --- Startup sequencing guards ---
        # Set of nodes whose parameter load has not yet completed.
        self._pending_params: Set[str] = set()
        # Guards against re-triggering activation while a previous batch is in flight.
        self._activating: bool = False

        # --- Subscriptions & Publishers ---
        self._as_info_sub: Subscription = self.create_subscription(
            AsInfo,
            "/as_status",
            self.as_info_callback,
            10
        )

        # New global health subscription
        self._health_sub: Subscription = self.create_subscription(
            Heartbeat,
            "/health",
            self.health_callback,
            10
        )

        self._mission_info_pub: Publisher = self.create_publisher(
            MissionInfo,
            "/system/mission_status",
            10
        )

        # --- Timers ---
        # 10Hz monitor loop for publishing status and checking heartbeats
        self._monitor_timer: Timer = self.create_timer(0.1, self.monitor_system)

      

        # Initialize tracking structures for the defined nodes
        self._init_node_tracking()

    def _init_node_tracking(self) -> None:
        """Sets up lifecycle clients, parameter clients and tracking data for monitored nodes."""
        for node in self.nodes_to_monitor:
            self._master_report[node] = {}
            self._node_heartbeats[node] = self.get_clock().now()

            self._lifecycle_clients[node] = self.create_client(
                ChangeState,
                f'{node}/change_state'
            )
            self._param_clients[node] = AsyncParameterClient(self, node)

    def _load_mission_parameters(self) -> None:
        """Loads the mission YAML to all monitored nodes via the standard parameter client.

        Runs fully asynchronously: once every node has acknowledged its
        parameters the CONFIGURE transition is triggered.
        """
        # Resolve the mission enum from the stored AsInfo message to a config name
        mission_name = mission_names.get(self._as_info.mission, "")

        if not mission_name:
            self.get_logger().error("No mission selected in AsInfo message. Cannot load parameters.")
            self.fail_mission()
            return

        try:
            share_dir = get_package_share_directory('mirena_supervisor')
        except Exception as e:
            self.get_logger().error(f"Could not find package share directory: {e}")
            self.fail_mission()
            return

        config_path = os.path.join(share_dir, 'mission_config', f"{mission_name}.yaml")

        if not os.path.exists(config_path):
            self.get_logger().error(f"Mission config file not found: {config_path}")
            self.fail_mission()
            return

        self.get_logger().info(f"Loading parameters from {config_path}...")

        # Reset lap counting for the new mission. Missions absent from the table
        # have a goal of 0, which disables lap-based completion.
        

        # Build the request set fully before dispatching, otherwise a fast
        # response could fire its callback and find _pending_params still
        # incomplete, advancing the sequence prematurely.
        self._pending_params = set()
        dispatch: List[tuple] = []
        for node in self.nodes_to_monitor:
            try:
                param_dict = parameter_dict_from_yaml_file(config_path, target_nodes=[node])
            except RuntimeError:
                # No section for this node in the config: nothing to load.
                self.get_logger().info(f"No parameters for '{node}' in {mission_name} config. Skipping.")
                continue

            params = list(param_dict.values())
            if not params:
                continue

            client = self._param_clients[node]
            if not client.wait_for_services(timeout_sec=1.0):
                self.get_logger().error(f"Parameter service for '{node}' not available.")
                self.fail_mission()
                return

            self._pending_params.add(node)
            dispatch.append((node, params, client))

        if not self._pending_params:
            self.get_logger().info("No parameters to load. Triggering CONFIGURE transition...")
            self._configure_nodes()
            return

        for node, params, client in dispatch:
            future = client.set_parameters(params)
            future.add_done_callback(lambda f, n=node: self._on_param_loaded(n, f))

    def _on_param_loaded(self, node_name: str, future: Future) -> None:
        """Handles a single node's parameter-set response and advances when all are done."""
        # Sequence may have been aborted/reset while this was in flight.
        if self._mission_info.status != MissionInfo.MISSION_STATUS_CONFIGURING:
            return

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Parameter load failed for '{node_name}': {e}")
            self.fail_mission()
            return

        if not all(result.successful for result in response.results):
            reasons = "; ".join(r.reason for r in response.results if not r.successful)
            self.get_logger().error(f"Parameters rejected by '{node_name}': {reasons}")
            self.fail_mission()
            return

        self.get_logger().info(f"Parameters loaded for '{node_name}'.")
        self._pending_params.discard(node_name)

        if not self._pending_params:
            self.get_logger().info("All parameters loaded. Triggering CONFIGURE transition...")
            self._configure_nodes()

    def _configure_nodes(self) -> None:
        """Triggers CONFIGURE on every monitored node; advances to READY when all succeed."""
        self._batch_transition(Transition.TRANSITION_CONFIGURE, self._on_all_configured)

    def _on_all_configured(self) -> None:
        if self._mission_info.status != MissionInfo.MISSION_STATUS_CONFIGURING:
            return
        self.get_logger().info("All nodes configured. Mission READY.")
        self._mission_info.status = MissionInfo.MISSION_STATUS_READY

    def _on_all_activated(self) -> None:
        if self._mission_info.status != MissionInfo.MISSION_STATUS_READY:
            return
        self.get_logger().info("All nodes activated. Mission RUNNING.")
        self._mission_info.status = MissionInfo.MISSION_STATUS_RUNNING
        self._activating = False

    def _batch_transition(
        self,
        transition_id: int,
        on_complete: Optional[Callable[[], None]] = None,
        critical: bool = True,
        reverse: bool = False
    ) -> None:
        """Requests a lifecycle transition on all monitored nodes.

        ``on_complete`` is invoked once every node has acknowledged success.
        When ``critical`` is False a failing transition is only logged (used for
        best-effort cleanup), otherwise it aborts the mission.
        """
        pending: Set[str] = set(self.nodes_to_monitor)
        if reverse == False:
            for node in self.nodes_to_monitor:
                self.get_logger().info(f"{node} called to transition")
                self._request_transition(node, transition_id, pending, on_complete, critical)
        else:
            for node in reversed(self.nodes_to_monitor):
                self.get_logger().info(f"{node} called to transition")
                self._request_transition(node, transition_id, pending, on_complete, critical)

    def _request_transition(
        self,
        node_name: str,
        transition_id: int,
        pending: Set[str],
        on_complete: Optional[Callable[[], None]],
        critical: bool,
    ) -> None:
        """Asynchronously requests a single lifecycle state transition without blocking the executor."""
        client: Optional[Client] = self._lifecycle_clients.get(node_name)
        if not client:
            self.get_logger().error(f"No lifecycle client found for {node_name}")
            if critical:
                self.fail_mission()
            return

        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f"Lifecycle service for {node_name} not available")
            if critical:
                self.fail_mission()
            return

        req = ChangeState.Request()
        req.transition.id = transition_id

        def callback(future: Future) -> None:
            try:
                result = future.result()
                ok = result is not None and result.success
            except Exception as e:
                self.get_logger().error(f"Service call failed for {node_name}: {e}")
                ok = False

            if not ok:
                self.get_logger().error(f"Failed transition {transition_id} for {node_name}")
                if critical:
                    self.fail_mission()
                    return

            pending.discard(node_name)
            if not pending and on_complete is not None:
                on_complete()

        client.call_async(req).add_done_callback(callback)

    def as_info_callback(self, msg: AsInfo) -> None:
        
        """Handles low-level Autonomous System state changes."""
        prev_status = self._as_info.status
        # Store the entire incoming message
        self._as_info = msg

        # Trigger configuring when AS goes to READY
        if self._as_info.status == AsInfo.AS_STATUS_READY and prev_status != AsInfo.AS_STATUS_READY:
            if self._mission_info.status in [MissionInfo.MISSION_STATUS_INACTIVE, MissionInfo.MISSION_STATUS_FAILED]:
                self.get_logger().info("AS Status is READY. Preparing configuration...")
                # Enter CONFIGURING immediately: this guards against re-entry and
                # the rest of the bring-up (params -> CONFIGURE -> READY) runs
                # asynchronously from here.
                self._mission_info.status = MissionInfo.MISSION_STATUS_CONFIGURING
                self._load_mission_parameters()

        # Trigger activate when AS goes to DRIVING
        elif self._as_info.status == AsInfo.AS_STATUS_DRIVING and self._mission_info.status == MissionInfo.MISSION_STATUS_READY:
            # READY persists until activation actually completes, so guard against
            # re-triggering the batch on repeated DRIVING messages.
            if not self._activating:
                self.get_logger().info("AS Status is DRIVING. Activating nodes...")
                self._activating = True
                self._batch_transition(Transition.TRANSITION_ACTIVATE, self._on_all_activated)

        elif self._as_info.status == AsInfo.AS_STATUS_EMERGENCY:
            self.get_logger().error("Emergency triggered from AS_STATUS!")
            self.fail_mission()

    def health_callback(self, msg: Heartbeat) -> None:
        """Aggregates node JSON reports and tracks timestamp from the global /health topic."""
        if self._mission_info.status != MissionInfo.MISSION_STATUS_RUNNING:
            return        
        # Ignore heartbeats from nodes we don't care about
        if msg.node_name not in self.nodes_to_monitor:
            return

        # Update last seen timestamp
        stamp_time = Time.from_msg(msg.timestamp)
        self._node_heartbeats[msg.node_name] = stamp_time

        # Update the master dictionary with the node's JSON payload
        try:
            report_data: Dict[str, Any] = json.loads(msg.report)
            self._master_report[msg.node_name] = report_data
        except json.JSONDecodeError:
            self.get_logger().warning(f"Malformed JSON report received from {msg.node_name}")

    def monitor_system(self) -> None:
        #self.get_logger().info(f"MONITORING!! {self._mission_info.status}")
        
        """Checks for dead nodes and publishes the current MissionStatus."""
        # 1. Check Node Heartbeats (Only enforce if we are active/configured)
        if self._mission_info.status not in [MissionInfo.MISSION_STATUS_READY, MissionInfo.MISSION_STATUS_RUNNING]:
            return

        now = self.get_clock().now()

        for node_name, last_report in self._node_heartbeats.items():
            elapsed_sec = (now - last_report).nanoseconds / 1e9

            if elapsed_sec > MAX_REPORT_SLACK_SEC:
                self.get_logger().error(
                    f"Critical Node '{node_name}' exceeded heartbeat threshold! "
                    f"({elapsed_sec:.2f}s > {MAX_REPORT_SLACK_SEC}s). Shutting down."
                )
                self.fail_mission()
                break


        # 3. Publish the full Mission Status message
        self._mission_info_pub.publish(self._mission_info)
        self._last_lap()
        self._slam_elapsed()

 

    def _last_lap(self) -> Optional[tuple]:
        """Latest start/finish gate (x, y) from the planning heartbeat report.

        Returns None until the planner has published a start point.
        """
        report = self._master_report.get(CONTROL_NODE_NAME, {})
        last_lap = report.get('last_lap')
       
        if last_lap == True :
            self.finish_mission()

    def _slam_elapsed(self) -> Optional[tuple]:
            """Latest start/finish gate (x, y) from the planning heartbeat report.
    
            Returns None until the planner has published a start point.
            """
            report = self._master_report.get('mirena_slam', {})
            elapsed_time = report.get('last_update_ms_elapsed')
            self.get_logger().info(
                        f"Time from last inference {elapsed_time}"
                    )
           
            
    


    

    def finish_mission(self) -> None:
        """Marks the mission FINISHED and brings nodes down (lap goal reached)."""
        if self._mission_info.status == MissionInfo.MISSION_STATUS_FINISHED:
            return  # Already finished

        self.get_logger().info(
            f"Lap goal reached. Mission FINISHED."
        )
        self._mission_info.status = MissionInfo.MISSION_STATUS_FINISHED

        # Reset any in-flight startup sequencing state.
        self._activating = False
        self._pending_params = set()

        # Best-effort deactivate then cleanup; errors here are non-critical.
        self._batch_transition(Transition.TRANSITION_DEACTIVATE, critical=False, reverse = True)
        self._batch_transition(Transition.TRANSITION_CLEANUP, critical=False, reverse = True)

    def fail_mission(self) -> None:
        """Sets mission to FAILED and attempts to deactivate/cleanup nodes."""
        if self._mission_info.status == MissionInfo.MISSION_STATUS_FAILED:
            return  # Already failed

        self.get_logger().fatal("ABORTING MISSION: System entered FAILED state.")
        self._mission_info.status = MissionInfo.MISSION_STATUS_FAILED

        # Reset any in-flight startup sequencing state.
        self._activating = False
        self._pending_params = set()

        # Best-effort deactivate then cleanup. Errors are non-critical here so a
        # node that is already inactive/unconfigured does not re-trigger failure.
        self._batch_transition(Transition.TRANSITION_DEACTIVATE, critical=False, reverse = True)
        self._batch_transition(Transition.TRANSITION_CLEANUP, critical=False, reverse = True)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = MirenaSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
