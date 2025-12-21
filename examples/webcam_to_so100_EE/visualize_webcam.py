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
from scipy.spatial.transform import Rotation

# Ensure we can import from the current directory if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.robot_utils import precise_sleep
from webcam_leader import WebcamLeader

FPS = 30

def main():
    print("Initializing Webcam Visualization...")
    
    # Initialize the Webcam Leader (try camera 0, 1, etc if needed)
    leader = WebcamLeader(camera_index=0)
    
    # Setup Kinematics for SO-100
    # We assume the urdf is available at "./SO101/so101_new_calib.urdf" relative to project root
    urdf_path = "SO101/so101_new_calib.urdf"
    if not os.path.exists(urdf_path):
        if os.path.exists("../../SO101/so101_new_calib.urdf"):
            urdf_path = "../../SO101/so101_new_calib.urdf"
        else:
            print(f"Warning: URDF not found at {urdf_path}. Joint calculation will fail.")
            return

    # Motor names for SO-100 (excluding gripper for IK usually, but listed in robot)
    # The IK solver typically expects the chain joints.
    # Based on SO100Follower class:
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    # Note: RobotKinematics usually needs the joints that are part of the kinematic chain.
    # Gripper is usually the end effector, so the chain goes up to wrist_roll.
    # Let's see what happens.
    
    try:
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=motor_names,
        )
    except Exception as e:
        print(f"Failed to initialize kinematics: {e}")
        return

    # Initial joint state (all zeros)
    # 6 motors
    current_joints = np.zeros(6) 
    
    # Connect to webcam
    leader.connect()
    
    print("Starting visualization loop... Press ESC in the window to exit.")
    
    try:
        while True:
            t0 = time.perf_counter()
            
            # Get EE action and image from webcam
            action_result = leader.get_action()
            if action_result is None:
                continue
            ee_action, image = action_result
            
            if ee_action is None or image is None:
                continue

            # Parse EE action
            x = ee_action["ee.x"]
            y = ee_action["ee.y"]
            z = ee_action["ee.z"]
            wx = ee_action["ee.wx"]
            wy = ee_action["ee.wy"]
            wz = ee_action["ee.wz"]
            gripper_pos = ee_action["ee.gripper_pos"]

            # Build target transform
            t_des = np.eye(4, dtype=float)
            t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            t_des[:3, 3] = [x, y, z]

            # Run Inverse Kinematics
            try:
                q_target = kinematics.inverse_kinematics(current_joints, t_des)
                current_joints = q_target
                
                # Overlay data on the image
                text_color = (0, 255, 0) # Green
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                line_height = 30
                
                cv2.putText(image, f"EE X: {x:.2f} m", (10, line_height), font, font_scale, text_color, thickness)
                cv2.putText(image, f"EE Y: {y:.2f} m", (10, line_height * 2), font, font_scale, text_color, thickness)
                cv2.putText(image, f"EE Z: {z:.2f} m", (10, line_height * 3), font, font_scale, text_color, thickness)
                cv2.putText(image, f"Gripper: {gripper_pos:.1f}", (10, line_height * 4), font, font_scale, text_color, thickness)
                
                # Display joint angles (up to 6 for SO100)
                for i, q in enumerate(q_target[:6]): # Limit to 6 joints
                    cv2.putText(image, f"J{i}: {q:.2f} deg", (10, line_height * (5 + i)), font, font_scale, text_color, thickness)
                
            except Exception as e:
                cv2.putText(image, f"IK Error: {e}", (10, image.shape[0] - 30), font, font_scale, (0, 0, 255), thickness)

            # Show the augmented image
            cv2.imshow('Webcam Visualization', image)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        leader.disconnect()

if __name__ == "__main__":
    main()
