#!/usr/bin/env python3
"""
Simple test for robot arm forward kinematics.
"""

import math
import sys
from pathlib import Path

# Add parent directory to path to import main module
sys.path.insert(0, str(Path(__file__).parent))

from main import forward_kinematics, LINK1_LENGTH, LINK2_LENGTH


def test_forward_kinematics() -> None:
    """Test forward kinematics with known values."""
    # Test case 1: Both joints at 0 degrees
    theta1 = 0.0
    theta2 = 0.0
    base_pos, joint1_pos, joint2_pos = forward_kinematics(theta1, theta2)

    # Joint 1 should be at (LINK1_LENGTH, 0, BASE_HEIGHT)
    expected_joint1 = (LINK1_LENGTH, 0.0, 0.1)
    assert abs(joint1_pos[0] - expected_joint1[0]) < 1e-6, f"Joint1 X: {joint1_pos[0]} != {expected_joint1[0]}"
    assert abs(joint1_pos[1] - expected_joint1[1]) < 1e-6, f"Joint1 Y: {joint1_pos[1]} != {expected_joint1[1]}"
    assert abs(joint1_pos[2] - expected_joint1[2]) < 1e-6, f"Joint1 Z: {joint1_pos[2]} != {expected_joint1[2]}"

    # End effector should be at (LINK1_LENGTH + LINK2_LENGTH, 0, BASE_HEIGHT)
    expected_ee = (LINK1_LENGTH + LINK2_LENGTH, 0.0, 0.1)
    assert abs(joint2_pos[0] - expected_ee[0]) < 1e-6, f"EE X: {joint2_pos[0]} != {expected_ee[0]}"
    assert abs(joint2_pos[1] - expected_ee[1]) < 1e-6, f"EE Y: {joint2_pos[1]} != {expected_ee[1]}"
    assert abs(joint2_pos[2] - expected_ee[2]) < 1e-6, f"EE Z: {joint2_pos[2]} != {expected_ee[2]}"

    # Test case 2: First joint at 90 degrees, second at 0
    theta1 = math.pi / 2
    theta2 = 0.0
    base_pos, joint1_pos, joint2_pos = forward_kinematics(theta1, theta2)

    # Joint 1 should be at (0, LINK1_LENGTH, BASE_HEIGHT)
    expected_joint1 = (0.0, LINK1_LENGTH, 0.1)
    assert abs(joint1_pos[0] - expected_joint1[0]) < 1e-6, f"Joint1 X: {joint1_pos[0]} != {expected_joint1[0]}"
    assert abs(joint1_pos[1] - expected_joint1[1]) < 1e-6, f"Joint1 Y: {joint1_pos[1]} != {expected_joint1[1]}"

    # Test case 3: Both joints at 90 degrees
    theta1 = math.pi / 2
    theta2 = math.pi / 2
    base_pos, joint1_pos, joint2_pos = forward_kinematics(theta1, theta2)

    # End effector should be at (0, LINK1_LENGTH + LINK2_LENGTH, BASE_HEIGHT)
    expected_ee = (0.0, LINK1_LENGTH + LINK2_LENGTH, 0.1)
    assert abs(joint2_pos[0] - expected_ee[0]) < 1e-6, f"EE X: {joint2_pos[0]} != {expected_ee[0]}"
    assert abs(joint2_pos[1] - expected_ee[1]) < 1e-6, f"EE Y: {joint2_pos[1]} != {expected_ee[1]}"

    print("✓ All forward kinematics tests passed!")


if __name__ == "__main__":
    test_forward_kinematics()
    print("All tests passed!")


