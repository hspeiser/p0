#!/usr/bin/env python3
"""
A simple robot arm simulation demo for Le Robot integration.

This example demonstrates:
- Forward kinematics for a 2-DOF robot arm
- Joint angle animation
- End-effector trajectory visualization
- Easy integration with Le Robot simulator data
"""

from __future__ import annotations

import argparse
import math
from typing import Final

import numpy as np
import rerun as rr

# Robot arm parameters
BASE_HEIGHT: Final = 0.1
LINK1_LENGTH: Final = 0.5
LINK2_LENGTH: Final = 0.4
JOINT_RADIUS: Final = 0.03
LINK_RADIUS: Final = 0.02


def forward_kinematics(theta1: float, theta2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute forward kinematics for a 2-DOF planar arm.

    Returns:
        base_pos: Base position (always at origin)
        joint1_pos: Position of first joint
        joint2_pos: Position of second joint (end effector)
    """
    # Base is at origin
    base_pos = np.array([0.0, 0.0, BASE_HEIGHT])

    # First joint position
    joint1_pos = base_pos + np.array([
        LINK1_LENGTH * math.cos(theta1),
        LINK1_LENGTH * math.sin(theta1),
        0.0
    ])

    # End effector position (second joint)
    joint2_pos = joint1_pos + np.array([
        LINK2_LENGTH * math.cos(theta1 + theta2),
        LINK2_LENGTH * math.sin(theta1 + theta2),
        0.0
    ])

    return base_pos, joint1_pos, joint2_pos


def log_robot_arm(
    theta1: float,
    theta2: float,
    trajectory: list[np.ndarray] | None = None,
) -> None:
    """Log the robot arm state to Rerun."""
    base_pos, joint1_pos, joint2_pos = forward_kinematics(theta1, theta2)

    # Log base
    rr.log(
        "world/robot/base",
        rr.Boxes3D(
            half_sizes=[[0.05, 0.05, BASE_HEIGHT]],
            centers=[base_pos],
            colors=[[100, 100, 200, 255]],
        ),
    )

    # Log first link (from base to joint1)
    link1_vector = joint1_pos - base_pos
    rr.log(
        "world/robot/link1",
        rr.Arrows3D(
            vectors=[link1_vector],
            origins=[base_pos],
            colors=[[200, 100, 100, 255]],
            radii=[LINK_RADIUS],
        ),
    )

    # Log first joint
    rr.log(
        "world/robot/joint1",
        rr.Boxes3D(
            half_sizes=[[JOINT_RADIUS, JOINT_RADIUS, JOINT_RADIUS]],
            centers=[joint1_pos],
            colors=[[255, 200, 100, 255]],
        ),
    )

    # Log second link (from joint1 to end effector)
    link2_vector = joint2_pos - joint1_pos
    rr.log(
        "world/robot/link2",
        rr.Arrows3D(
            vectors=[link2_vector],
            origins=[joint1_pos],
            colors=[[100, 200, 100, 255]],
            radii=[LINK_RADIUS],
        ),
    )

    # Log end effector
    rr.log(
        "world/robot/end_effector",
        rr.Boxes3D(
            half_sizes=[[JOINT_RADIUS * 1.5, JOINT_RADIUS * 1.5, JOINT_RADIUS * 1.5]],
            centers=[joint2_pos],
            colors=[[255, 255, 0, 255]],
        ),
    )

    # Log trajectory if provided
    if trajectory:
        rr.log(
            "world/robot/trajectory",
            rr.LineStrips3D(
                [trajectory],
                colors=[[255, 200, 0, 255]],
                radii=[0.005],
            ),
        )


def generate_trajectory(num_points: int = 100) -> list[tuple[float, float]]:
    """
    Generate a smooth trajectory for the robot arm.

    Returns a list of (theta1, theta2) joint angles.
    """
    trajectory = []
    for i in range(num_points):
        t = i / num_points * 2 * math.pi

        # Create a figure-8 pattern
        theta1 = math.pi / 3 + 0.5 * math.sin(t)
        theta2 = math.pi / 2 + 0.8 * math.sin(2 * t)

        trajectory.append((theta1, theta2))

    return trajectory


def simulate_robot_arm(
    steps: int = 1000,
    trajectory: list[tuple[float, float]] | None = None,
) -> None:
    """Run the robot arm simulation."""
    # Set up coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Generate trajectory if not provided
    if trajectory is None:
        trajectory = generate_trajectory(steps)

    # Store end effector positions for trajectory visualization
    ee_trajectory: list[np.ndarray] = []

    for step, (theta1, theta2) in enumerate(trajectory[:steps]):
        # Use set_time with a sequence timeline (set_time_sequence deprecated)
        rr.set_time("step", sequence=step)

        # Compute forward kinematics
        base_pos, joint1_pos, joint2_pos = forward_kinematics(theta1, theta2)

        # Store end effector position
        ee_trajectory.append(joint2_pos.copy())

        # Log current state
        log_robot_arm(theta1, theta2, ee_trajectory if len(ee_trajectory) > 1 else None)

        # Log joint angles as text (useful for debugging)
        rr.log(
            "world/robot/joint_angles",
            rr.TextLog(
                f"θ1: {theta1:.2f} rad ({math.degrees(theta1):.1f}°), "
                f"θ2: {theta2:.2f} rad ({math.degrees(theta2):.1f}°)"
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple robot arm simulation demo for Le Robot integration",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--lerobot-mode",
        action="store_true",
        help="Enable Le Robot integration mode (ready for external joint angles)",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "rerun_example_robot_arm")

    if args.lerobot_mode:
        # In Le Robot mode, you would read joint angles from Le Robot simulator
        # For now, we'll use a simple trajectory
        print("Le Robot mode: Ready to receive joint angles from simulator")
        print("To integrate: Replace trajectory with Le Robot joint angle data")
        trajectory = generate_trajectory(args.steps)
    else:
        trajectory = None

    simulate_robot_arm(args.steps, trajectory)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()


