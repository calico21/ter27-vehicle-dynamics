from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def launch_setup(context, *args, **kwargs):
    """Executes at runtime when launch arguments have real string values."""
    
    # 1. Resolve the LaunchConfigurations into actual Python strings
    state_gt  = LaunchConfiguration('ground_truth_state').perform(context)
    percep_gt = LaunchConfiguration('ground_truth_perception').perform(context)
    slam_gt   = LaunchConfiguration('ground_truth_slam').perform(context)
    track_gt  = LaunchConfiguration('ground_truth_track').perform(context)

    # 2. Safely use standard Python logic to build the list of tuples
    dynamic_remappings = []

    # NOTE: ground_truth_state is handled by mirena_state's passthrough mode
    # (it republishes /debug/state/car onto /state/car), not by a remap here.
    if percep_gt == 'true':
        dynamic_remappings.append(( 'debug/perception', 'perception/cones'))
    if slam_gt == 'true':
        dynamic_remappings.append(( 'car/debug_slam', 'slam/map'))
    if track_gt == 'true':
        dynamic_remappings.append(( 'track_manager/track', 'planning/track'))

    # 3. Create the node with the dynamic list
    sim_node = Node(
        package='mirena_sim',
        executable='mirena_sim',        
        parameters=[{'use_sim_time': True}],
        remappings=dynamic_remappings
    )

    autonomy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('mirena_bringup'), 'launch', 'autonomy.launch.py'])
        ]),
        # Forward the ground-truth switches so the autonomy stack can gate
        # its nodes (previously only use_sim_time was forwarded).
        launch_arguments={
            'use_sim_time': 'true',
            'ground_truth_state': state_gt,
            'ground_truth_perception': percep_gt,
            'ground_truth_slam': slam_gt,
        }.items(),
    )

    # 4. Return the setup actions to be executed
    return [sim_node, autonomy]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mission', default_value='trackdrive'),
        DeclareLaunchArgument('ground_truth_state',       default_value='false'),
        DeclareLaunchArgument('ground_truth_perception',  default_value='false'),
        DeclareLaunchArgument('ground_truth_slam',        default_value='false'),
        DeclareLaunchArgument('ground_truth_track',       default_value='false'),

        # Pass control to the runtime function
        OpaqueFunction(function=launch_setup)
    ])