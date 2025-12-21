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
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

# Ensure we can import from the current directory if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.robot_utils import precise_sleep
from webcam_leader import WebcamLeader
from imu_reader import IMUReader

FPS = 30
IMU_PORT = "/dev/tty.usbmodem2101"

def main():
    print("Initializing 6-DOF Mimic Simulation with IMU...")
    
    # Initialize Rerun
    rr.init("SO-100 6-DOF Mimic", spawn=True)

    # Initialize IMU
    imu = IMUReader(port=IMU_PORT)
    try:
        imu.connect()
    except Exception as e:
        print(f"Warning: Could not connect to IMU at {IMU_PORT}: {e}")
        print("Check permissions (EPERM) or if device is plugged in.")
        print("Continuing without IMU (Wrist control will use fallback/locked values).")
        imu = None

    # Initialize the Webcam Leader with IMU
    leader = WebcamLeader(camera_index=0, imu_reader=imu)
    
    # Setup Kinematics for SO-100 (Still needed for Forward Kinematics visualization)
    urdf_path = "SO101/so101_new_calib.urdf"
    if not os.path.exists(urdf_path):
        if os.path.exists("../../SO101/so101_new_calib.urdf"):
            urdf_path = "../../SO101/so101_new_calib.urdf"
        else:
            print(f"Warning: URDF not found at {urdf_path}. Simulation will fail.")
            if imu: imu.disconnect()
            return

    # IMPORTANT: The joint_names list must EXACTLY match the order expected by the URDF for placo.
    # SO-100 typically has 5 arm joints + 1 gripper joint.
    # The order below matches the standard SO-100 configuration and our calculations.
    motor_names = [
        "shoulder_pan", 
        "shoulder_lift", 
        "elbow_flex", 
        "wrist_flex", 
        "wrist_roll", 
        "gripper" # Gripper is often a prismatic joint or a separate mechanism, but included here for consistency
    ]
    
    try:
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=motor_names,
        )
    except Exception as e:
        print(f"Failed to initialize kinematics: {e}")
        print("Ensure 'placo' is installed and URDF is valid.")
        if imu: imu.disconnect()
        return

    # Heuristic list of frames to visualize
    skeleton_chain = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "lower_arm_link",
        "wrist_link",
        "gripper_link",
        "gripper_frame_link"
    ]
    available_frames = list(kinematics.robot.frame_names())
    valid_skeleton = []
    for target in skeleton_chain:
        matches = [f for f in available_frames if target in f]
        if matches:
            valid_skeleton.append(matches[0])
            
    print(f"Visualizing Skeleton Chain: {valid_skeleton}")

    leader.connect()
    
    print("Starting 6-DOF mimic loop... Check the Rerun window for 3D view!")
    
    try:
        while True:
            t0 = time.perf_counter()
            
            # Get 6 Joint Angles and image from webcam
            joints_target, image = leader.get_action()
            
            if joints_target is None or image is None:
                continue

            # joints_target is now a numpy array of 6 elements: [J1..J5, Gripper]
            if len(joints_target) != 6:
                print(f"Error: Expected 6 joint values, got {len(joints_target)}")
                continue

            # Update FK with these joints (placo expects degrees)
            kinematics.forward_kinematics(joints_target)
            
            # --- VISUALIZATION (RERUN) ---
            
            # Extract positions/rotations for skeleton
            positions = []
            for frame_name in valid_skeleton:
                tf = kinematics.robot.get_T_world_frame(frame_name)
                pos = tf[:3, 3]
                rot = tf[:3, :3]
                positions.append(pos)
            
            if positions:
                rr.log("world/robot/skeleton", rr.LineStrips3D([positions], radii=0.005, colors=[200, 200, 200]))

            # Visualize Gripper
            gripper_val = joints_target[5]
            gripper_tf = kinematics.robot.get_T_world_frame("gripper_frame_link")
            g_pos = gripper_tf[:3, 3]
            g_rot = gripper_tf[:3, :3]
            half_width = (gripper_val / 100.0) * 0.05 # Map 0-100 to 0-0.05m half-width
            finger_1 = g_pos + g_rot[:, 1] * half_width # Assuming Y axis spreads fingers
            finger_2 = g_pos - g_rot[:, 1] * half_width
            
            rr.log("world/robot/gripper_fingers", rr.Points3D(
                [finger_1, finger_2],
                radii=0.01,
                colors=[[255, 255, 0], [255, 255, 0]],
                labels=["Left Finger", "Right Finger"]
            ))

            # Log joint angles to Rerun plot
            labels = ["J1_Pan", "J2_Lift", "J3_Elbow", "J4_W_Flex", "J5_W_Roll", "J6_Gripper"]
            for i, val in enumerate(joints_target):
                rr.log(f"plot/{labels[i]}", rr.Scalars(val))

            # --- OVERLAY TEXT ON WEBCAM ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "Mode: 6-DOF Mimic (Cam + IMU)", (10, 30), font, 0.7, (0, 255, 255), 2)
            
            # Display values for each joint
            display_labels = ["Pan", "Lift", "Elbow", "W-Flex", "W-Roll", "Grip"]
            for i, val in enumerate(joints_target):
                color = (0, 255, 0)
                if i == 5: color = (0, 255, 255) # Gripper is yellow
                if (i == 3 or i == 4) and imu: color = (255, 0, 255) # IMU driven joints are Magenta
                
                cv2.putText(image, f"{display_labels[i]}: {val:.1f} deg", (10, 60 + i*25), font, 0.6, color, 2)

            cv2.imshow('Webcam Leader', image)

            if cv2.waitKey(1) & 0xFF == 27:
                break
                
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        leader.disconnect()
        if imu:
            imu.disconnect()

if __name__ == "__main__":
    main()