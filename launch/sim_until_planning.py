import launch
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch the mirena_sim node with the --track argument
        #Node(
        #    package='mirena_sim',
        #    executable='sim',
        #    name='mirena_sim',
        #    output='screen',
        #    arguments=['--track=TrackFiles/track.json']
        #),
        
        # Launch the mirena_lidar node
        Node(
            package='mirena_lidar',
            executable='mirena_lidar',
            name='mirena_lidar',
            output='screen'
        ),

        
        Node(
            package='mirena_slam',
            executable='mirena_slam',
            name='mirena_slam',
            output='screen',
            remappings=[
                ("/debug/sim/perception_entities", "/perception/entities")#,
                #("/debug/sim/car_state", "/consensus/car_state")
            ]
        ),
        Node(
            package='mirena_planning',
            executable='mirena_planning',
            name='mirena_planning',
            output='screen',
            remappings=[
                # ("/debug/sim/car_state", "/consensus/car_state"),
                ("/debug/sim/slam_entities", "/slam/full_map")
            ]
        )
    ]

def main():
    # Create a launch description for the ROS 2 launch system
    launch_service = launch.LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    
    # Execute the launch service and keep it alive
    launch_service.run()

if __name__ == '__main__':
    main()

