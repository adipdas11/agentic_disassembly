"""Complete bringup for the laptop ReAct disassembly flow.

This launch owns one instance of every required runtime component:

* dual-arm MoveIt/EXOTica hardware stack
* hand-eye calibration publisher
* FT300 standalone driver
* Orbbec, tool camera, crop pipeline, and split vision inference nodes
* vision dashboard/debug feed
* screwdriver controller
* UF850-only Vuer Quest teleoperation (no xArm5 teleoperation conflict)
* laptop ReAct agent
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _default_moveit_servo_setup() -> str:
    candidates = [
        os.environ.get("MOVEIT_SERVO_SETUP", "").strip(),
        "/home/adip/workspace/dev_ws/install/moveit_servo/share/moveit_servo/local_setup.bash",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), "")


def generate_launch_description():
    disassembly_share = Path(get_package_share_directory("disassembly_skill"))
    arm_teleop_share = Path(get_package_share_directory("arm_teleop"))

    hardware_type = LaunchConfiguration("hardware_type")
    tool_video_device = LaunchConfiguration("tool_video_device")
    vision_backend = LaunchConfiguration("vision_backend")
    moveit_servo_setup = LaunchConfiguration("moveit_servo_setup")
    launch_uf850_teleop = LaunchConfiguration("launch_uf850_teleop")
    teleop_config = LaunchConfiguration("teleop_config")
    vuer_config = LaunchConfiguration("vuer_config")
    vuer_host = LaunchConfiguration("vuer_host")
    vuer_port = LaunchConfiguration("vuer_port")
    vuer_cert_file = LaunchConfiguration("vuer_cert_file")
    vuer_key_file = LaunchConfiguration("vuer_key_file")
    enable_camera_streams = LaunchConfiguration("enable_camera_streams")
    teleop_start_delay = LaunchConfiguration("teleop_start_delay")
    agent_start_delay = LaunchConfiguration("agent_start_delay")
    vision_timeout_s = LaunchConfiguration("vision_timeout_s")

    base_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(disassembly_share / "launch" / "disassembly_system.launch.py")
        ),
        launch_arguments={
            "hardware_type": hardware_type,
            "tool_video_device": tool_video_device,
            "vision_backend": vision_backend,
            "moveit_servo_setup": moveit_servo_setup,
        }.items(),
    )

    # Importing vuer_quest_teleop.launch.py would start a second MoveIt stack.
    # Bring up only its runtime nodes and reuse the base system instead.
    vuer_bridge = Node(
        package="arm_teleop",
        executable="vuer_quest_bridge",
        name="vuer_quest_bridge",
        output="screen",
        parameters=[
            vuer_config,
            {
                "host": vuer_host,
                "port": ParameterValue(vuer_port, value_type=int),
                "cert_file": vuer_cert_file,
                "key_file": vuer_key_file,
                "enable_camera_streams": ParameterValue(
                    enable_camera_streams,
                    value_type=bool,
                ),
                "enable_camera1_stream": False,
                "enable_camera2_stream": False,
                "enable_hand_tracking": True,
            },
        ],
    )

    wait_for_exotica = Node(
        package="arm_teleop",
        executable="wait_for_exotica_ready",
        name="wait_for_exotica_ready",
        output="screen",
        parameters=[{"timeout_sec": 240.0}],
    )

    uf850_teleop = Node(
        package="arm_teleop",
        executable="exotica_arm_teleop",
        name="exotica_arm_teleop",
        output="screen",
        parameters=[
            teleop_config,
            {
                "hardware_type": hardware_type,
                "enable_uf850": True,
                "enable_xarm5": False,
                "uf850.hand": "right",
                "xarm5.hand": "left",
                "uf850.track_orientation": False,
                "uf850.track_yaw": True,
                "uf850.track_pitch": True,
                "uf850.track_roll": True,
            },
        ],
    )

    teleop_panel = Node(
        package="arm_teleop",
        executable="teleop_control_panel",
        name="teleop_control_panel",
        output="screen",
    )

    teleop_runtime = GroupAction(
        condition=IfCondition(launch_uf850_teleop),
        actions=[
            LogInfo(msg="EXOTica is ready; starting UF850-only teleoperation"),
            uf850_teleop,
            teleop_panel,
        ],
    )

    laptop_agent = Node(
        package="disassembly_skill",
        executable="laptop_auto_agent",
        name="laptop_auto_agent",
        output="screen",
        parameters=[{"vision_timeout_s": vision_timeout_s}],
    )

    def start_ready_consumers(event, _context):
        if event.returncode != 0:
            return [
                LogInfo(
                    msg=(
                        "EXOTica readiness check failed; teleoperation and the "
                        "disassembly agent will not start."
                    )
                )
            ]
        return [
            teleop_runtime,
            LogInfo(msg="Starting laptop ReAct agent after EXOTica became ready"),
            laptop_agent,
        ]

    readiness_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_exotica,
            on_exit=start_ready_consumers,
        )
    )

    readiness_gate = TimerAction(
        period=teleop_start_delay,
        actions=[
            GroupAction(
                condition=IfCondition(launch_uf850_teleop),
                actions=[
                    LogInfo(msg="Starting Vuer Quest bridge"),
                    vuer_bridge,
                ],
            ),
            wait_for_exotica,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("hardware_type", default_value="real"),
            DeclareLaunchArgument(
                "tool_video_device",
                default_value="auto",
                description="Tool camera device or auto for stable device discovery",
            ),
            DeclareLaunchArgument(
                "vision_backend",
                default_value="rfdetr",
                description="Vision inference backend passed to the base system",
            ),
            DeclareLaunchArgument(
                "moveit_servo_setup",
                default_value=_default_moveit_servo_setup(),
                description="Optional MoveIt Servo overlay setup file",
            ),
            DeclareLaunchArgument(
                "launch_uf850_teleop",
                default_value="true",
                description="Start Vuer Quest tracking and UF850-only teleoperation",
            ),
            DeclareLaunchArgument(
                "teleop_config",
                default_value=str(
                    arm_teleop_share / "config" / "quest_exotica_teleop.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "vuer_config",
                default_value=str(
                    arm_teleop_share / "config" / "vuer_quest_bridge.yaml"
                ),
            ),
            DeclareLaunchArgument("vuer_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("vuer_port", default_value="8012"),
            DeclareLaunchArgument(
                "vuer_cert_file",
                default_value=EnvironmentVariable("VUER_CERT_FILE", default_value=""),
            ),
            DeclareLaunchArgument(
                "vuer_key_file",
                default_value=EnvironmentVariable("VUER_KEY_FILE", default_value=""),
            ),
            DeclareLaunchArgument(
                "enable_camera_streams",
                default_value="false",
                description="Stream camera panels to Quest; disabled for disassembly bringup",
            ),
            DeclareLaunchArgument(
                "teleop_start_delay",
                default_value="12.0",
                description="Seconds before starting UF850 teleoperation nodes",
            ),
            DeclareLaunchArgument(
                "agent_start_delay",
                default_value="30.0",
                description=(
                    "Deprecated compatibility argument; the agent now starts only "
                    "after /exotica/ready"
                ),
            ),
            DeclareLaunchArgument(
                "vision_timeout_s",
                default_value="180.0",
                description="Maximum time the agent waits for each exact screw label",
            ),
            LogInfo(msg="Starting complete laptop disassembly system"),
            base_system,
            readiness_handler,
            readiness_gate,
        ]
    )

