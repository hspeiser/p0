#!/usr/bin/env python
"""Gamepad teleoperation with end-effector control for SO-101."""

import time
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
ROBOT_PORT = "/dev/tty.usbmodem5AB90670321"

# End-effector step sizes (how much to move per joystick input)
EE_STEP_SIZES = {"x": 0.002, "y": 0.002, "z": 0.002}

# End-effector bounds (workspace limits in meters)
EE_BOUNDS = {"min": [-0.5, -0.5, -0.1], "max": [0.5, 0.5, 0.5]}


def main():
    # Initialize robot config
    follower_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="so101_follower",
        use_degrees=True
    )

    # Initialize gamepad config
    gamepad_config = GamepadTeleopConfig(use_gripper=True)

    # Initialize robot and gamepad
    follower = SO101Follower(follower_config)
    gamepad = GamepadTeleop(gamepad_config)

    # Connect
    follower.connect()
    gamepad.connect()

    # Get motor names (excluding gripper for IK)
    motor_names = list(follower.bus.motors.keys())
    ik_motor_names = [n for n in motor_names if n != "gripper"]

    print(f"Motor names: {motor_names}")
    print(f"IK motor names: {ik_motor_names}")

    # Initialize kinematics solver
    kinematics = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=ik_motor_names,
    )

    # Build processor pipeline: gamepad deltas -> EE pose -> joint positions
    ee_reference = EEReferenceAndDelta(
        kinematics=kinematics,
        end_effector_step_sizes=EE_STEP_SIZES,
        motor_names=ik_motor_names,
        use_latched_reference=False,
    )

    ee_bounds = EEBoundsAndSafety(
        end_effector_bounds=EE_BOUNDS,
        max_ee_step_m=0.05,
    )

    gripper_vel_to_pos = GripperVelocityToJoint(
        speed_factor=20.0,
        clip_min=0.0,
        clip_max=100.0,
        discrete_gripper=True,
    )

    ik_solver = InverseKinematicsEEToJoints(
        kinematics=kinematics,
        motor_names=motor_names,
        initial_guess_current_joints=True,
    )

    pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ee_reference, gripper_vel_to_pos, ee_bounds, ik_solver],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    print("\n" + "="*50)
    print("Gamepad Teleoperation with End-Effector Control")
    print("="*50)
    print("Controls:")
    print("  Left stick  : Move X/Y")
    print("  Right stick : Move Z (up/down)")
    print("  RB          : Close gripper")
    print("  LT          : Open gripper")
    print("  Ctrl+C      : Exit")
    print("="*50 + "\n")

    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            obs = follower.get_observation()

            # Get gamepad action (delta_x, delta_y, delta_z, gripper)
            raw_action = gamepad.get_action()

            # Convert gamepad action to the format expected by processors
            teleop_action = {
                "enabled": True,
                "target_x": raw_action["delta_x"],
                "target_y": raw_action["delta_y"],
                "target_z": raw_action["delta_z"],
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper_vel": raw_action.get("gripper", 1),  # 0=close, 1=stay, 2=open
            }

            # Process through pipeline: deltas -> EE pose -> joint positions
            joint_action = pipeline((teleop_action, obs))

            # Send to robot
            follower.send_action(joint_action)

            # Display
            print(f"\rGamepad: x={raw_action['delta_x']:+.2f} y={raw_action['delta_y']:+.2f} z={raw_action['delta_z']:+.2f} | ", end="")
            print(f"FPS: {1/(time.perf_counter()-t0):.0f}  ", end="", flush=True)

            precise_sleep(max(1.0/FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        gamepad.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
