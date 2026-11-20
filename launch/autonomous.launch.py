from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package = 'my_car_description_2'
    share = Path(get_package_share_directory(package))
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', EnvironmentVariable(
            'FASTRTPS_DEFAULT_PROFILES_FILE', default_value=str(share / 'config/fastdds.xml'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/sim.launch.py')),
            launch_arguments={
                'world': 'autonomy_course.sdf',
                'ekf': 'true',
                'x': '0', 'y': '0', 'yaw': '0',
                'rviz_config': 'course.rviz',
                'headless': LaunchConfiguration('headless'),
                'rviz': LaunchConfiguration('rviz'),
            }.items(),
        ),
        Node(package=package, executable='drive_to_goal',
             parameters=[{'use_sim_time': True}], output='screen'),
    ])
