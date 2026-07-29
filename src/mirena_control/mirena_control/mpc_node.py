import math
import numpy as np
import threading
import traceback
import json
from typing import Optional

from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from mirena_control.mpc.mpc_controller import MPCController

# ROS2 Messages
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from mirena_common.msg import Track, Car, CarControl, Heartbeat, AsInfo
from std_msgs.msg import Header
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn 

# Transform handling
import tf2_ros
from rclpy.duration import Duration

# MPC Hyperparameters
MPC_TRACK_POINTS = 20
MPC_STEPS = 20
MPC_T = 2.0 
MPC_DT = MPC_T / MPC_STEPS
LOOKAHEAD_BASE = 15.0
LOOKAHEAD_GAIN = 1.2
ROBOT_FRAME = 'car/cog'  # The car body frame (used to look up the car pose)
WORK_FRAME = 'odom' 

# How close (metres) the car must come to the start/finish gate for a crossing to count
LAP_ZONE_RADIUS_M: float = 5.0

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

class MPCNode(LifecycleNode):
    def __init__(self):
        super().__init__('MPCNode')
        self.trajectory: Optional[CubicSpline] = None
        self.car: Optional[Car] = None
        self.start_gate: Optional[np.ndarray] = None
        self._as_info: Optional[AsInfo] = None
        self.is_track_closed = False
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.global_s = 0.0          
        self.state_window = np.zeros((MPC_STEPS + 1, 10))
        self.action_window = np.zeros((MPC_STEPS, 3))
        self.ref_window = np.zeros((MPC_TRACK_POINTS, 2))
        
        self.previous_pos = None
        self.lock = threading.Lock()
        self.mpc = None

        # -- Lap counting ---
        self.lap_count: int = 0
        self.lap_goal: int = 0
        self._lap_in_zone: bool = False
        self._lap_armed: bool = True
        self.last_lap = False
        self.last_solve_start_time = None
        self.finish = False

    def as_info_callback(self, msg: AsInfo):
        """Callback to store autonomous system status and mission."""
        self._as_info = msg
        mission_name = mission_names.get(msg.mission, "")
        if mission_name in mission_laps:
            self.lap_goal = mission_laps[mission_name]

    def on_configure(self, state: rclpy.lifecycle.State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring Local-Frame Control Node...")
        try:
            self.last_solve_start_time = self.get_clock().now()
            self.mpc = MPCController(path_points=MPC_TRACK_POINTS, N=MPC_STEPS, T=MPC_T)
            
            mission_name = mission_names.get(self._as_info.mission, "") if self._as_info else ""
            
            self.mpc_cb_group = MutuallyExclusiveCallbackGroup()
            self.ctrl_cb_group = MutuallyExclusiveCallbackGroup()

            self.track_sub = self.create_subscription(Track, 'planning/track', self.track_callback, 10)
            self.car_sub = self.create_subscription(Car, 'state/car', self.car_callback, 10)
            self.mission_sub = self.create_subscription(
                AsInfo,
                "/as_status",
                self.as_info_callback,
                10
            )
            self.control_pub = self.create_lifecycle_publisher(CarControl, '/control', 10)
            self.viz_pub = self.create_lifecycle_publisher(Marker, 'control/horizon_markers', 10)
            self.heartbeat_pub = self.create_lifecycle_publisher(Heartbeat, 'health', 10)
            
            self.heartbeat_timer = self.create_timer(0.1, self.publish_heartbeat)
            self.mpc_timer = self.create_timer(0.05, self.mpc_iteration, callback_group=self.mpc_cb_group)  
            self.control_timer = self.create_timer(0.01, self.control_executor, callback_group=self.ctrl_cb_group)

            self.lap_goal = mission_laps.get(mission_name, 0)
            self.lap_count = 0
            self._lap_in_zone = True
            self._lap_armed = False

            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Failed to configure MPCNode: {e}\n{traceback.format_exc()}")
            return TransitionCallbackReturn.FAILURE

    # --- 1. DATA CALLBACKS ---
    def track_callback(self, msg: Track):
        if not msg.gates:
            return
        points = np.array([(g.x, g.y) for g in msg.gates])
        self.start_gate = points[0]
        if len(points) < 2: return
        self.is_track_closed = msg.is_closed
        bc = 'periodic' if self.is_track_closed else 'not-a-knot'
        if self.is_track_closed: points = np.vstack((points, points[0]))
        arc_lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        self.trajectory = CubicSpline(arc_lengths, points, bc_type=bc)

    def car_callback(self, msg: Car):
        if not self.control_pub.is_activated: return
        with self.lock:
            self.car = msg

    def _update_lap_count(self, car) -> None:
 

        if car is None or self.start_gate is None:
            return

        dist = math.hypot(car[0] - self.start_gate[0], car[1] - self.start_gate[1])

        if dist > LAP_ZONE_RADIUS_M:
            self._lap_in_zone = False
            self._lap_armed = True
        elif self._lap_armed and not self._lap_in_zone:
            self._lap_in_zone = True
            self._lap_armed = False
            self.lap_count += 1
            self.get_logger().info(f"Lap {self.lap_count}/{self.lap_goal} completed.")
            if self.lap_count == self.lap_goal:
                self.last_lap = True


    @staticmethod
    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _transform_points_2d(points, transform):
        t = transform.transform.translation
        yaw = MPCNode._yaw_from_quat(transform.transform.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        return points @ R.T + np.array([t.x, t.y])
    
    # --- 2. LOCAL REFERENCE GENERATION ---
    def generate_reference(self):
        if not self.trajectory or not self.car:
            return None

        target_time = self.car.header.stamp

        try:
            if not self.tf_buffer.can_transform('map', ROBOT_FRAME, target_time, timeout=Duration(seconds=0.005)):
                return None

            t_m2c = self.tf_buffer.lookup_transform('map', ROBOT_FRAME, target_time)
            car_pos_map = np.array([t_m2c.transform.translation.x, t_m2c.transform.translation.y])

            t_o2c = self.tf_buffer.lookup_transform(WORK_FRAME, ROBOT_FRAME, target_time)
            car_pose_odom = np.array([
                t_o2c.transform.translation.x,
                t_o2c.transform.translation.y,
                self._yaw_from_quat(t_o2c.transform.rotation),
            ])
            
            s_max = self.trajectory.x[-1]
            if self.last_lap and self.global_s >= 30:
                self.finish = True


            is_closed = self.is_track_closed and not self.last_lap 
            def dist_sq(s):
                s_ev = s[0] % s_max if is_closed else np.clip(s[0], 0, s_max)
                return np.sum((self.trajectory(s_ev) - car_pos_map)**2)

            res = minimize(dist_sq, x0=np.array([self.global_s]), method='Nelder-Mead')
            self.previous_pos = self.global_s
            self.global_s = float(res.x[0])

            lookahead_dist = LOOKAHEAD_BASE + (self.car.u * LOOKAHEAD_GAIN)
            s_query = np.linspace(self.global_s, self.global_s + lookahead_dist, MPC_TRACK_POINTS)
            s_eval = s_query % s_max if self.is_track_closed else np.clip(s_query, 0, s_max)
            path_map = self.trajectory(s_eval)

            t_o2m = self.tf_buffer.lookup_transform(WORK_FRAME, 'map', target_time)
            path_odom = self._transform_points_2d(path_map, t_o2m)

            return path_odom, lookahead_dist, car_pose_odom

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF Lookup failed in generate_reference: {e}", throttle_duration_sec=2.0)
            return None

    # --- 3. DIAGNOSTICS ---
    def log_solver_diagnostics(self, status, end_time):
        internal_time = self.mpc.ocp_solver.get_stats('time_tot') * 1000 
        sqp_iters = self.mpc.ocp_solver.get_stats('sqp_iter')
        res = self.mpc.ocp_solver.get_stats('residuals')
        max_res = np.max(res) if res is not None else 0.0
        
        total_lat_ms = (end_time - Time.from_msg(self.car.header.stamp)).nanoseconds / 1e6 if self.car else 0.0
        
        if status == 0:
            self.get_logger().info(
                f"LOCAL OK | Solve: {internal_time:4.1f}ms | Lat: {total_lat_ms:4.1f}ms | "
                f"Iters: {sqp_iters} | Res: {max_res:.1e}", throttle_duration_sec=0.5
            )
        else:
            self.get_logger().error(f"MPC FAIL Status: {status}")

    # --- 4. CORE MPC LOOP ---
    def mpc_iteration(self):
        if not self.control_pub.is_activated:
            return
        
        ref_data = self.generate_reference()
        if ref_data is None: return
        self.ref_window, path_len, car_pose = ref_data
        
        with self.lock:
            if self.car is None: return
            x0 = np.zeros(self.mpc.model.x.shape[0])
            x0[0:3] = car_pose
            x0[3:6] = [self.car.u, self.car.v, self.car.omega]
            x0[6:8] = self.state_window[1, 6:8]
            x0[8], x0[9] = 0.0, self.car.u
            
            state_warm, action_warm = self.state_window.copy(), self.action_window.copy()

        params = np.hstack([self.ref_window[:, 0], self.ref_window[:, 1], np.array([path_len])])

        t_start = self.get_clock().now()
        status, x_opt, u_opt = self.mpc.run(x0, state_warm, action_warm, params)
        t_end = self.get_clock().now()
        
        if status == 0 and x_opt is not None:
            with self.lock:
                self.state_window, self.action_window = x_opt, u_opt
                self.last_solve_start_time = t_start
        else:
            self.mpc.ocp_solver.reset()

        self.log_solver_diagnostics(status, t_end)
        self.broadcast_debug()
        self._update_lap_count(car_pose)

    # --- 5. CONTROL & VIZ ---
    def control_executor(self):
        if not self.control_pub.is_activated: 
            return

        if self.last_lap:
            ctrl_msg = CarControl(gas=0.0, steer_angle=0.0)
            ctrl_msg.header.stamp = self.get_clock().now().to_msg()
            self.control_pub.publish(ctrl_msg)
            return

        with self.lock:
            if not self.car or self.last_solve_start_time is None: return
            dt = (self.get_clock().now() - self.last_solve_start_time).nanoseconds / 1e9
            idx = max(0, min(int(dt / MPC_DT), MPC_STEPS - 1))
            gas, steer = float(self.state_window[idx, 6]), float(self.state_window[idx, 7])
        
        ctrl_msg = CarControl(gas=gas, steer_angle=steer)
        ctrl_msg.header.stamp = self.get_clock().now().to_msg()
        self.control_pub.publish(ctrl_msg)



    def broadcast_debug(self):
        with self.lock:
            r, p = self.ref_window.copy(), self.state_window[:, 0:2].copy()
        self.viz_pub.publish(self._create_marker(r, "local_ref", (1.0, 0.0, 0.0), 0))
        self.viz_pub.publish(self._create_marker(p, "local_pred", (0.0, 1.0, 1.0), 1))

    def _create_marker(self, p_list, ns, rgb, m_id):
        m = Marker(header=Header(frame_id=WORK_FRAME, stamp=self.get_clock().now().to_msg()),
                   ns=ns, id=m_id, type=Marker.LINE_STRIP, action=Marker.ADD)
        m.scale.x, m.color.r, m.color.g, m.color.b, m.color.a = 0.1, rgb[0], rgb[1], rgb[2], 1.0
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.1) for p in p_list]
        return m

    def publish_heartbeat(self):
        if self.heartbeat_pub and self.heartbeat_pub.is_activated:
            msg = Heartbeat()
            msg.timestamp = self.get_clock().now().to_msg()
            msg.node_name = self.get_name()

            report_data = {
                "status_state": "WORKING",
                "progress": self.global_s,
                "last_lap": self.finish
            }
            msg.report = json.dumps(report_data)
            self.heartbeat_pub.publish(msg)

    def on_activate(self, state):
        self.get_logger().info("Activating MPCNode timers...")
        # Re-enable timers
        if hasattr(self, 'mpc_timer'):
            self.mpc_timer.reset()
        if hasattr(self, 'control_timer'):
            self.control_timer.reset()
        super().on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info("Deactivating MPCNode...")

        if hasattr(self, 'mpc_timer'):
            self.mpc_timer.cancel()
        if hasattr(self, 'control_timer'):
            self.control_timer.cancel()

        with self.lock:
            if self.mpc and hasattr(self.mpc, 'ocp_solver'):
                self.mpc.ocp_solver.reset()
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        super().on_cleanup(state)
        self.get_logger().info('Cleaning up node resources...')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()