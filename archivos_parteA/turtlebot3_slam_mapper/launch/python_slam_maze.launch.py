#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot3_slam_mapper')
    turtlebot3_gazebo_pkg_share = get_package_share_directory('turtlebot3_gazebo')

    # Launch arguments from turtlebot3_maze.launch.py
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-0.45')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    yaw = LaunchConfiguration('yaw', default='-1.57079632679')

    # RViz configuration (can reuse the one from slam_toolbox setup)
    rviz_config_file = LaunchConfiguration('rviz_config', 
                                           default=os.path.join(pkg_share, 'rviz', 'slam_maze.rviz'))

    # Gazebo simulation (turtlebot3_maze.launch.py)
    gazebo_launch_file = os.path.join(turtlebot3_gazebo_pkg_share, 'launch', 'turtlebot3_maze.launch.py')
    
    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'x_pose': x_pose,
            'y_pose': y_pose,
            'yaw': yaw,
            'turtlebot_type': 'burger'
        }.items()
    )

    # Our Python SLAM Node
    python_slam_node = Node(
        package='turtlebot3_slam_mapper',
        executable='python_slam_node',
        name='python_slam_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            # Add any specific parameters for python_slam_node if needed here
            # e.g., {'map_resolution': 0.1}
            # For now, it will use defaults or parameters set in its own code.
        ]
    )

    # RViz2
    start_rviz2_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file])

    # Teleop Twist Keyboard (optional, if xterm is installed and you want to use it)
    # start_teleop_twist_keyboard_node = Node(
    #     package='teleop_twist_keyboard',
    #     executable='teleop_twist_keyboard',
    #     name='teleop_twist_keyboard',
    #     output='screen',
    #     prefix='xterm -e' # Launch in a new terminal
    # )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch arguments
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation (Gazebo) clock'))
    ld.add_action(DeclareLaunchArgument('x_pose', default_value='-0.45', description='Initial x pose of the robot'))
    ld.add_action(DeclareLaunchArgument('y_pose', default_value='0.0', description='Initial y pose of the robot'))
    ld.add_action(DeclareLaunchArgument('yaw', default_value='-1.57079632679', description='Initial yaw of the robot'))
    ld.add_action(DeclareLaunchArgument('rviz_config', default_value=os.path.join(pkg_share, 'rviz', 'slam_maze.rviz'), description='Full path to the RVIZ config file to use'))
    
    # Add actions to LaunchDescription
    ld.add_action(start_gazebo_cmd)
    ld.add_action(python_slam_node) # Add our Python SLAM node
    ld.add_action(start_rviz2_cmd)
    # ld.add_action(start_teleop_twist_keyboard_node) # Uncomment if you want teleop

    return ld 