from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'language',
            default_value='en',
            description='Language code for the ASR model'
        ),
        DeclareLaunchArgument(
            'model',
            default_value='base',
            description='Model type for the ASR model'
        ),
        DeclareLaunchArgument(
            'silence_grace_period',
            default_value='1.0',
            description='Grace period in seconds after silence to stop recording'
        ),
        DeclareLaunchArgument(
            'sample_rate',
            default_value='0',
            description='Sample rate for audio capture (0 for auto-detect)'
        ),
        Node(
            package='asr',  # Replace with your actual package name
            executable='asr_node',  # Replace with your node's executable name
            name='automatic_speech_recognition',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'language': LaunchConfiguration('language'),
                'model': LaunchConfiguration('model'),
                'silence_grace_period': LaunchConfiguration('silence_grace_period'),
                'sample_rate': LaunchConfiguration('sample_rate'),
            }],
        )
    ])