import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node, LifecycleNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # --- 1. CONFIGURATION ---
    # Paths (adjust if using ament_index_python for portability)
    track_dir = "/home/mirena/mirena_ws/src/mirena_sim/TrackFiles/"
    rviz_config = "/home/mirena/mirena_ws/launch/full.rviz"
    
    common_params = {'use_sim_time': False}

    print(os.getcwd())
    #1. Setup Paths
    nodes_to_supervise = [
        'mirena_slam',
        'mirena_planning',
        'control_node'
    ]

    supervisor_node = Node(
        package='mirena_supervisor', 
        executable='mirena_supervisor', 
        name='mirena_supervisor',
        output='screen',
        parameters=[
            *common_params,
            {'nodes_to_monitor': nodes_to_supervise} # Passing the string list parameter
        ]
    )

    #args = [
    #    DeclareLaunchArgument('mode', default_value='sim', description='sim or real'),
    #    DeclareLaunchArgument('track', default_value='random', description='skidpad, acceleration, or random'),
    #    DeclareLaunchArgument('use_debug_perception', default_value='true'),
    #]
    #mode = LaunchConfiguration('mode')
    #track = LaunchConfiguration('track')
    #use_debug_perception = LaunchConfiguration('use_debug_perception')
    common_params = [{'use_sim_time': True}] # Usually True for simulation
#
    ## --- 2. CONDITIONAL MAPPING ---
    ## Convert track name to full path for the simulator
    #track_path = PythonExpression([
    #    f"'{track_dir}' + '{track}.json' if '", track, "' != 'random' else 'random'"
    #])
    supervisor_node = Node(
        package='mirena_supervisor', 
        executable='mirena_supervisor', 
        name='mirena_supervisor',
        output='screen',
        parameters=[*common_params]
    )

    sim_node = Node(
        package='mirena_sim', 
        executable='mirena_sim',
        parameters=[*common_params],
    )

    # 2. Define the Perception Lifecycle Node
    # Explicitly using keyword arguments to satisfy the 'namespace' requirement

    slam_node = LifecycleNode(
        package='mirena_slam',
        executable='mirena_slam',
        name = 'mirena_slam',
        namespace='',
        output='screen',
        
    )
    planning_node = LifecycleNode(
        package='mirena_planning',
        executable='mirena_planning',
        name = 'mirena_planning',
        namespace='',
        output='screen',
        remappings=[
                ('car/debug_slam', '/slam/map')
            ],
        
    )
    control_node = LifecycleNode(
        package='mirena_control',
        executable='control_node',
        name = 'mirena_control',
        namespace='',
        output='screen',
       
        
    )
    rviz_node = Node(
        package='rviz2', 
        executable='rviz2', 
        name='rviz2',
        parameters=common_params,
        arguments=['-d', rviz_config]
        
    )





    return LaunchDescription([
        supervisor_node,
        control_node,
        rviz_node,
        sim_node,
        slam_node,
        planning_node,
        
    ])
