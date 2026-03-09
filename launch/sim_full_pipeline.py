import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Define the parameter once to reuse it
    common_params = [{'use_sim_time': False}]

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            parameters=common_params,
            # Optional: Path to your .rviz config file
            arguments=['-d', "/home/mirena/mirena_ws/launch/full.rviz"] 
            
        ),
        Node(
            package='mirena_sim',
            executable='mirena_sim',
            name='mirena_sim',
            output='log',
            parameters=common_params,
            arguments = ['--random']
        ),
        
        Node(
            package='mirena_perception',
            executable='perception_node',
            name='mirena_perception',
            output='log',
            parameters=common_params
        ),
        
        Node(
            package='mirena_slam',
            executable='mirena_slam',
            name='mirena_slam',
            output='screen',
            parameters=common_params,
            remappings=[
                ("/debug/sim/perception_entities", "/perception/entities")
            ]
        ),
        
        Node(
            package='mirena_planning',
            executable='planning_node',
            name='mirena_planning',
            output='log',
            parameters=common_params,
            remappings=[("/sim/debug/slam", "/slam/full_map")]
        ),

        Node(
            package='mirena_control',
            executable='control_node',
            name='mirena_control',
            output='log',
            remappings=[("/sim/debug/track", "/planner/track")
                        ,("/sim/debug/car", "/slam/car_state")
                        ],
            parameters=common_params
        )
    ])

def main():
    launch_service = launch.LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()

if __name__ == '__main__':
    main()