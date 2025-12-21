#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import cv2
import sys
import os

# Ensure we can import from the current directory if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep

# Import our local WebcamLeader
from webcam_leader import WebcamLeader

FPS = 30


def main():
    # Initialize the robot config
    # Adjust the port as necessary for your system
    # You can find the port using `lerobot-find-port` or `ls /dev/tty.*`
    follower_port = "/dev/tty.usbmodem5A460814411" # CHANGE THIS TO YOUR PORT
    
    # Try to find a valid port if the default doesn't look right or user hasn't set it
    # For now, we'll assume the user might need to edit this or pass it as arg
    # But let's try to be smart? No, hardcoding for the example structure is standard, 
    # but I'll add a check or comment.
    
    print(f"Using Follower Port: {follower_port}")
    print("If this is incorrect, please edit the script or ensure the device is connected.")

    follower_config = SO100FollowerConfig(
        port=follower_port, id="my_webcam_controlled_arm", use_degrees=True
    )

    # Initialize the robot
    follower = SO100Follower(follower_config)

    # Initialize the Webcam Leader
    leader = WebcamLeader()

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo
    # We assume the urdf is available at "./SO101/so101_new_calib.urdf" relative to project root
    urdf_path = "SO101/so101_new_calib.urdf"
    if not os.path.exists(urdf_path):
        # fallback to absolute path or check if we are in examples dir
        if os.path.exists("../../SO101/so101_new_calib.urdf"):
            urdf_path = "../../SO101/so101_new_calib.urdf"
        else:
            print(f"Warning: URDF not found at {urdf_path}. Kinematics might fail.")

    follower_kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    # build pipeline to convert EE action to robot joints
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.3, -0.3, 0.05], "max": [0.3, 0.3, 0.4]}, # Adjusted safe bounds
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    follower.connect()
    leader.connect()

    print("Starting teleop loop... Press ESC in the webcam window to exit.")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = follower.get_observation()

            # Get teleop action (EE pose) from webcam
            leader_ee_act = leader.get_action()
            
            if leader_ee_act is None:
                continue

            # teleop EE -> robot joints
            try:
                follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))
                
                # Send action to robot
                _ = follower.send_action(follower_joints_act)
            except ValueError as e:
                print(f"Kinematics/Safety Error: {e}")
            except Exception as e:
                print(f"Error: {e}")

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
