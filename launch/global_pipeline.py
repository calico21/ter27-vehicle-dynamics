import launch
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PythonExpression, EqualsSubstitution
from launch.conditions import LaunchConfigurationEquals, IfCondition, UnlessCondition
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import LifecycleNode

def generate_launch_description():
    # 1. Declare Launch Arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='sim',
        description='Operating mode: "sim" or "car"'
    )

    mission_arg = DeclareLaunchArgument(
        'mission',
        default_value='random',
        description='Track type: "dicovery", "skidpad" or "acceleration"'
    )

    # 2. Get the values from the configurations
    mode = LaunchConfiguration('mode')
    mission = LaunchConfiguration('mission')
    common_params = [{'use_sim_time': False}]

    # 3. Dynamic Map Path Selection Logic
    map_path_expr = PythonExpression([
        "'/home/mirena/mirena_ws/src/mirena_sim/TrackFiles/skidpad.json' if '", mission, "' == 'skidpad' else (",
        "'/home/mirena/mirena_ws/src/mirena_sim/TrackFiles/acceleration.json' if '", mission, "' == 'acceleration' else 'random')"
    ])

    # --- NODE DEFINITIONS ---
    # Note: Removed trailing commas to prevent them from becoming tuples
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        parameters=common_params,
        arguments=['-d', "/home/mirena/mirena_ws/launch/full.rviz"]
    )

    sim_node = Node(
        condition=IfCondition(PythonExpression(["'", mode, "' == 'sim'"])),
        package='mirena_sim',
        executable='mirena_sim',
        output='screen',
        parameters=[
            *common_params,
            {'track': map_path_expr}
        ], 
    )
    
    perception_node = Node(
        package='mirena_perception',
        executable='perception_node',
        name='mirena_perception',
        output='log',
        parameters=common_params,
      
    )
    
    slam_node = Node(
        package='mirena_slam',
        executable='mirena_slam',
        name='mirena_slam',
        output='screen',
        parameters=common_params,
    )

    planning_node = LifecycleNode(
        package ='mirena_planning',
        executable = 'planning_node',
        name='mirena_planning',
        namespace='',
        output = 'screen',
        parameters=[*common_params, {'mission': mission}, {'lap_count': 1}]
    )

    control_node = Node(
        package='mirena_control',
        executable='control_node',
        name='mirena_control',
        output='log',
        parameters=common_params
    )

    supervisor_node = Node(
        package='mirena_supervisor',
        executable='supervisor_node',
        name='mirena_supervisor',
        output='screen',
        parameters=common_params
    )

    nodes_to_namespace = [
        sim_node, 
        perception_node, 
        slam_node, 
        planning_node,
        control_node,
        supervisor_node
    ]

    # This pushes everything inside the list into the namespace defined by 'mode'
    namespaced_group = GroupAction(
        actions=[
            #PushRosNamespace(mode),
            *nodes_to_namespace
        ]
    )



    return LaunchDescription([
        mode_arg,
        mission_arg,
        rviz_node,         
        namespaced_group,  
    ])

def main():
    launch_service = launch.LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()

if __name__ == '__main__':
    main()