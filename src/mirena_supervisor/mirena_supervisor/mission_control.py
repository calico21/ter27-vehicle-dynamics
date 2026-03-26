import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition, State
from enum import Enum

class MissionStatus(Enum):
    IDLE = 0 
    CONFIGURING = 1
    MISSION_ONGOING = 2
    MISSION_FINISHED = 3
    MISSION_FAILED = 4


class MissionControlNode(Node):
    def __init__(self):
        super().__init__('supervisor_node')
        

        self.nodes = ['mirena_perception', 'mirena_slam', 'mirena_planning', 'mirena_control']
        self.heartbeat_topics = {
            'mirena_perception': 'perception/heartbeat',
            'mirena_slam': 'slam/heartbeat',
            'mirena_planning': 'planner/heartbeat',
            'mirena_control': 'control/heartbeat'
        }
        
        self.timeout_threshold = 1
        self.heartbeat_timer = self.create_timer(0.5, self.check_heartbeats)
        self.status: MissionStatus = MissionStatus.IDLE
        self.last_heartbeats = {node: self.get_clock().now() for node in self.nodes}
        self.node_active = {node: False for node in self.nodes}

        self.client_nodes = {node: self.create_client(ChangeState, f'{node}/change_state') for node in self.nodes}
        
        self.get_logger().info("Starting sequential boot: Configuring all nodes first...")
        self.run_sequence()

    def run_sequence(self):
        self.configure_all()

    def configure_all(self):
        self.pending_configures = len(self.nodes)
        self.status = MissionStatus.CONFIGURING
        for node in self.nodes:
            self.send_transition(node, Transition.TRANSITION_CONFIGURE, self.check_all_configured)

    def check_all_configured(self, future, node_name):
        res = future.result()
        if res.success:
            self.get_logger().info(f"Successfully CONFIGURED {node_name}")
            self.pending_configures -= 1
            if self.pending_configures == 0:
                self.get_logger().info("All nodes Configured. Now ACTIVATING...")
                self.activate_all()
        else:
            self.get_logger().error(f"Failed to configure {node_name}!")

    def activate_all(self):
        for node in self.nodes:
            self.send_transition(node, Transition.TRANSITION_ACTIVATE, self.on_activate_done)

    def on_activate_done(self, future, node_name):
        res = future.result()
        if res.success:
            self.get_logger().info(f"Successfully ACTIVATED {node_name}")
            self.create_subscription(
                Header, 
                self.heartbeat_topics[node_name], 
                lambda msg, n=node_name: self.heartbeat_callback(msg, n), 
                10)
            self.status = MissionStatus.MISSION_ONGOING
        else:
            self.get_logger().error(f"Failed to activate {node_name}!")

    def send_transition(self, node_name, transition_id, callback):
        client = self.client_nodes[node_name]
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f"Service for {node_name} not respoding!")
            return

        req = ChangeState.Request()
        req.transition.id = transition_id
        future = client.call_async(req)
        future.add_done_callback(lambda f, n=node_name: callback(f, n))

    def heartbeat_callback(self, msg, node_name):
        self.last_heartbeats[node_name] = self.get_clock().now()
        self.node_active[node_name] = True

    def check_heartbeats(self):

        if self.status != MissionStatus.MISSION_ONGOING:
            return

        now = self.get_clock().now()

        for node_name in self.nodes:

            last_time = self.last_heartbeats[node_name]
            
            elapsed_sec = (now - last_time).nanoseconds / 1e9

            if elapsed_sec > self.timeout_threshold:
                self.get_logger().error(
                    f"CRITICAL: Heartbeat lost from {node_name}! "
                    f"Last seen {elapsed_sec:.2f}s ago."
                )
                
                self.status = MissionStatus.MISSION_FAILED
                
                self.node_active[node_name] = False
                self.handle_mission_failure(node_name)
            
    def handle_mission_failure(self):
        
        for node in self.nodes:
            self.send_transition(node, Transition.TRANSITION_ON_SHUTDOWN_ERROR, self.on_shutdown)
    
    def on_shutdown(self):
        self.get_logger().error(f"FAILURE: SHUTDOWN")



def main(args=None):
    rclpy.init(args=args)
    node = MissionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()