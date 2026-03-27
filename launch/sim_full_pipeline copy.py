import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import LaunchConfigurationEquals

MAP_LOOKUP = {
    "skidpad": "/home/anadigar/mirena_ws/src/mirena_planning/resource/skidpad.json",
    "accel":   "/home/anadigar/mirena_ws/src/mirena_planning/resource/acceleration.json"
}
common_params = {'use_sim_time': False}

def launch_setup(context, *args, **kwargs):
    code_word = LaunchConfiguration('map_name').perform(context)
    

    resolved_path = MAP_LOOKUP.get(code_word, code_word)

    planning_mode = LaunchConfiguration('planning_mode').perform(context)

    known_planning_node = Node(
        package='mirena_planning',
        executable='known_planning',
        name='mirena_planning',
        condition=LaunchConfigurationEquals('mission', 'known'),
        parameters=[common_params, {'track_path': resolved_path}],
        remappings=[("/sim/debug/slam", "/slam/full_map")]
    )

    return [known_planning_node]

def generate_launch_description():

    map_choice = LaunchConfiguration('map_name')
    return LaunchDescription([
        # The argument now take a "code word"
        DeclareLaunchArgument(
            'map_name', 
            default_value='skidpad',
            description='Code word for the map: skidpad, random, track or accel'
        ),
        
        DeclareLaunchArgument(
            'mission',
            default_value='track',
            description='Options: track, discovery, known'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            parameters=[common_params],
            arguments=['-d', "/home/mirena/mirena_ws/launch/full.rviz"] 
        ),
        
        Node(
            package='mirena_sim',
            executable='mirena_sim',
            name='mirena_sim',
            parameters=[common_params],
            arguments=['--', map_choice]
        ),
        
        Node(
            package='mirena_perception',
            executable='perception_node',
            name='mirena_perception',
            parameters=[common_params]
        ),
        
        Node(
            package='mirena_slam',
            executable='mirena_slam',
            name='mirena_slam',
            output='screen',
            parameters=[common_params],
            remappings=[("/debug/sim/perception_entities", "/perception/entities")]
        ),
        

        Node(
            package='mirena_planning',
            executable='discovery_planning',
            name='mirena_planning',
            condition=LaunchConfigurationEquals('planning_mode', 'discovery'),
            parameters=[common_params],
            remappings=[("/sim/debug/slam", "/slam/full_map")]
        ),

        

        Node(
            package='mirena_control',
            executable='control_node',
            name='mirena_control',
            remappings=[
                ("/sim/debug/track", "/planner/track"),
                ("/sim/debug/car", "/slam/car_state")
            ],
            parameters=[common_params]
        ),

        OpaqueFunction(function=launch_setup)
    ])