import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node, LifecycleNode

def generate_launch_description():
    # --- 1. CONFIGURATION ---
    # Paths (adjust if using ament_index_python for portability)
    track_dir = "/home/mirena/mirena_ws/src/mirena_sim/TrackFiles/"
    rviz_config = "/home/mirena/mirena_ws/launch/full.rviz"
    
    # Arguments
    args = [
        DeclareLaunchArgument('mode', default_value='sim', description='sim or real'),
        DeclareLaunchArgument('track', default_value='random', description='skidpad, acceleration, or random'),
        DeclareLaunchArgument('use_debug_perception', default_value='false'),
    ]

    # Substitutions
    mode = LaunchConfiguration('mode')
    track = LaunchConfiguration('track')
    use_debug_perception = LaunchConfiguration('use_debug_perception')
    common_params = [{'use_sim_time': True}] # Usually True for simulation

    # --- 2. CONDITIONAL MAPPING ---
    # Convert track name to full path for the simulator
    track_path = PythonExpression([
        f"'{track_dir}' + '{track}.json' if '", track, "' != 'random' else 'random'"
    ])

    # Define which nodes the supervisor should manage
    nodes_to_supervise = PythonExpression([
        "['mirena_slam', 'mirena_planning', 'mirena_control'] if '", 
        use_debug_perception, 
        "' == 'false' else ['mirena_perception', 'mirena_slam', 'mirena_planning', 'mirena_control']"
    ])

    # Dynamic remapping for perception
    perception_topic = PythonExpression([
        "'car/debug_perception' if '", use_debug_perception, "' == 'true' else 'perception/entities'"
    ])
    consensus_topic = PythonExpression([
        "'car/debug_state' if '", use_debug_perception, "' == 'true' else 'consensus/car_state'"
    ])

    # --- 3. NODE DEFINITIONS ---

    # Simulator (Only node that receives the track file)
    sim_node = Node(
        condition=IfCondition(PythonExpression(["'", mode, "' == 'sim'"])),
        package='mirena_sim', 
        executable='mirena_sim',
        parameters=[*common_params, {'track': track_path}],
    )

    supervisor_node = Node(
        package='mirena_supervisor', 
        executable='supervisor_node', 
        name='mirena_supervisor',
        output='screen',
        parameters=[*common_params, {'nodes_to_supervise': nodes_to_supervise}]
    )

    # Runtime Nodes
    #consensus_node = Node(
    #    condition=UnlessCondition(use_debug_perception),
    #    package='mirena_consensus', 
    #    executable='mirena_consensus', 
    #    name='mirena_consensus',
    #    parameters=common_params,
    #)

    perception_node = Node(
        condition=UnlessCondition(use_debug_perception),
        package='mirena_perception', 
        executable='perception_node', 
        name='mirena_perception',
        parameters=common_params,
    )

    slam_node = Node(
        package='mirena_slam', 
        executable='mirena_slam', 
        name='mirena_slam',
        parameters=common_params,
        remappings=[('perception/entities', perception_topic), ('consensus/car_state', consensus_topic)]
    )

    planning_node = LifecycleNode(
        package='mirena_planning', 
        executable='planning_node', 
        name='mirena_planning',
        namespace='', 
        parameters=common_params
    )

    control_node = Node(
        package='mirena_control', 
        executable='control_node', 
        name='mirena_control',
        parameters=common_params,
        remappings=[('consensus/car_state', consensus_topic)]
    )

    rviz_node = Node(
        package='rviz2', 
        executable='rviz2', 
        name='rviz2',
        parameters=common_params,
        arguments=['-d', rviz_config]
    )

    # --- 4. EXECUTION ---
    return LaunchDescription(args + [
        rviz_node,
        sim_node,
        supervisor_node,
        perception_node,
        #consensus_node,
        slam_node,
        planning_node,
        control_node
    ])