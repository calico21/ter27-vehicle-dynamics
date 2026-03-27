import launch
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction # Used to bundle nodes into the namespace

def generate_launch_description():
    common_params = [{'use_sim_time': False}]
    
    # 1. Define your global namespace here
    namespace_name = '/sim/car'

    # 2. Wrap all nodes in a GroupAction
    nodes_with_namespace = GroupAction(
        actions=[
            # This action pushes the namespace to everything in this list
            PushRosNamespace(namespace_name),

            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                parameters=common_params,
                arguments=['-d', "/home/mirena/mirena_ws/launch/full.rviz"] 
            ),
            Node(
                package='mirena_sim',
                executable='mirena_sim',
                name='mirena_sim',
                parameters=common_params,
                arguments=['--random']
            ),
            Node(
                package='mirena_perception',
                executable='perception_node',
                name='mirena_perception',
                parameters=common_params
            ),
            Node(
                package='mirena_slam',
                executable='mirena_slam',
                name='mirena_slam',
                parameters=common_params,
            ),
            Node(
                package='mirena_planning',
                executable='discovery_planning',
                name='mirena_planning',
                parameters=common_params,
            ),
            Node(
                package='mirena_control',
                executable='control_node',
                name='mirena_control',

                parameters=common_params
            )
        ]
    )

    return LaunchDescription([nodes_with_namespace])

def main():
    launch_service = launch.LaunchService()
    launch_service.include_launch_description(generate_launch_description())
    launch_service.run()

if __name__ == '__main__':
    main()