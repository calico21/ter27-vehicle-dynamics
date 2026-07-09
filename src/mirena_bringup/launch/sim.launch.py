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

    if state_gt == 'true':
        dynamic_remappings.append(( 'car/debug_state', 'car/state'))
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
        launch_arguments={'use_sim_time': 'true'}.items(),
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