import launch
from launch import LaunchDescription
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
            package='mirena_perception',
            executable='perception_node',
            name='mirena_perception',
            output='screen'
        ),

        Node(
            package='mirena_consensus',
            executable='mirena_consensus',
            name='mirena_consensus',
            output='screen'
        ),
        
        Node(
            package='mirena_slam',
            executable='mirena_slam',
            name='mirena_slam',
            output='screen',
            remappings=[
                ("/debug/sim/perception_entities", "/perception/entities"),
                ("/debug/sim/car_state", "/consensus/car_state")
            ]
        )
    ])

def main():
    # Create a launch description for the ROS 2 launch system
    launch_service = launch.LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    
    # Execute the launch service and keep it alive
    launch_service.run()

if __name__ == '__main__':
    main()

