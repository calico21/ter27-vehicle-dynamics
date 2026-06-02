import os
import sys
import lifecycle_msgs.msg
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode
from launch.actions import EmitEvent, RegisterEventHandler
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import StaticTransforms

def generate_launch_description():
    print(os.getcwd())
    #1. Setup Paths
    hw_config = os.path.join(
        get_package_share_directory('mirena_hardware'),
        'config', 'camera.yaml'
    )

    # 2. Define the Perception Lifecycle Node
    # Explicitly using keyword arguments to satisfy the 'namespace' requirement
    perception_node = LifecycleNode(
        package='mirena_perception',
        executable='mirena_perception',
        name='mirena_perception',
        namespace='',  # <--- This must be a keyword argument
        output='screen',
        remappings=[
            ('car/camera/image_raw', 'car/camera/image_raw'),
            ('car/camera/camera_info', 'car/camera/camera_info'),
            ('car/lidar/cloud', 'car/lidar/cloud') 
        ]
    )

    camera_node = LifecycleNode(
        package='mirena_hardware',
        executable='mirena_camera',
        name = 'mirena_camera',
        namespace='',
        parameters=[hw_config],
        output='screen'
    )

    lidar_node = LifecycleNode(
        package='mirena_hardware',
        executable='mirena_lidar',
        name = 'mirena_lidar',
        namespace='',
        output='screen'
    )


    # 3. Transition Logic: Unconfigured -> Inactive (Configure)
    configure_perception = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda node: node == perception_node,
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    configure_camera = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda node: node == camera_node,
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    configure_lidar = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda node: node == lidar_node,
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    # 4. Transition Logic: Inactive -> Active (Activate)
    activate_camera = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=perception_node,
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=lambda node: node == camera_node,
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                )
            ]
        )
    )
    activate_lidar = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=perception_node,
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=lambda node: node == lidar_node,
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                )
            ]
        )
    )
    activate_perception = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=perception_node,
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=lambda node: node == perception_node,
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                    )
                )
            ]
        )
    )

    tf_nodes = [
            # LiDAR: x=0.6, y=0.05 (5cm Left), z=0.25
            Node(package='tf2_ros', executable='static_transform_publisher',
                name='tf_cog_lidar', 
                arguments=['-0.2', '-0.15', '0.25', '0', '0', '0', 'car/cog', 'car/lidar']),
            
            # Camera: x=0.6, y=-0.05 (5cm Right), z=0.25
            Node(package='tf2_ros', executable='static_transform_publisher',
                name='tf_cog_cam', 
                arguments=['0', '0.0', '0.25', '0', '0', '0', 'car/cog', 'car/camera_link']),
            
            # Optical rotation (Standard ROS camera convention: Z forward, X right, Y down)
            Node(package='tf2_ros', executable='static_transform_publisher',
                name='tf_cam_optical', 
                arguments=['0', '0', '0', '-1.571', '0', '-1.571', 'car/camera_link', 'car/camera_optical'])
            
    ]


    return LaunchDescription([
        *tf_nodes,
        camera_node,
        lidar_node,
        perception_node,
        configure_camera,
        configure_lidar,
        configure_perception,
        activate_camera,
        activate_lidar,
        activate_perception
    ])