import cv2
import numpy as np
import time
import threading
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

class PoseLandmark:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

class HandLandmark:
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_TIP = 8
    PINKY_MCP = 17

class WebcamLeader:
    def __init__(self, camera_index=0, width=1280, height=800, imu_reader=None):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.imu_reader = imu_reader
        self.cap = None
        self.running = False
        
        # Threading
        self.lock = threading.Lock()
        self.latest_frame = None       # Latest RGB frame from camera
        self.latest_result_img = None  # Processed image with overlay
        self.latest_joints = np.zeros(6)
        
        # MediaPipe Tasks
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        pose_model_path = os.path.join(model_dir, 'pose_landmarker.task')
        hand_model_path = os.path.join(model_dir, 'hand_landmarker.task')

        # Initialize Pose Landmarker
        base_options_pose = mp_tasks.BaseOptions(model_asset_path=pose_model_path)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

        # Initialize Hand Landmarker
        base_options_hand = mp_tasks.BaseOptions(model_asset_path=hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

        # Connections
        self.POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
        self.HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
        
        # Custom Drawing Specs for "More Color"
        self.hand_connection_color = (121, 22, 76)  # BGR
        self.hand_landmark_color = (250, 44, 250)   # BGR
        
        # Smoothing
        self.joints_smooth = np.zeros(6)
        self.alpha = 0.2

        # Timestamp tracking for MediaPipe (must be monotonically increasing)
        self._last_timestamp_ms = 0 

    def _draw_landmarks(self, image, landmarks, connections, landmark_color=(0, 255, 0), connection_color=(255, 0, 0), thickness=2, radius=2):
        if not landmarks:
            return
        h, w, _ = image.shape
        
        # Draw connections
        if connections:
            for start_idx, end_idx in connections:
                if start_idx >= len(landmarks) or end_idx >= len(landmarks): continue
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                # Check visibility/presence if available (also check for None)
                if hasattr(start, 'visibility') and start.visibility is not None and start.visibility < 0.5: continue
                if hasattr(end, 'visibility') and end.visibility is not None and end.visibility < 0.5: continue
                if hasattr(start, 'presence') and start.presence is not None and start.presence < 0.5: continue
                if hasattr(end, 'presence') and end.presence is not None and end.presence < 0.5: continue

                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(image, p1, p2, connection_color, thickness)
                
        # Draw landmarks
        for lm in landmarks:
            if hasattr(lm, 'visibility') and lm.visibility is not None and lm.visibility < 0.5: continue
            if hasattr(lm, 'presence') and lm.presence is not None and lm.presence < 0.5: continue
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(image, (px, py), radius, landmark_color, -1)

    def connect(self):
        print(f"Attempting to open camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            # Try fallback index 1 just in case
            print("Index 0 failed, trying index 1...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open webcam {self.camera_index} or 1")
        
        # Try High-Performance Settings (Arducam OV9782)
        # MJPG is crucial for high FPS over USB
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 100)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimal latency

        # Enable auto-exposure (reset from any previous manual settings)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto, 1 = manual
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
        
        # Verify what we actually got
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"Camera Initialized: {actual_w}x{actual_h} @ {actual_fps}fps (Codec: {codec_str})")
        
        self.running = True
        
        # Start Threads
        # 1. Capture Thread (Grabs frames as fast as possible)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # 2. Process Thread (Runs MediaPipe and Math)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

    def disconnect(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Webcam disconnected.")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            # Convert to RGB here to save time in process loop
            # Arducam MJPG might need RB swap? Original code did cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with self.lock:
                self.latest_frame = frame_rgb

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: return v
        return v / norm

    def _get_angle_from_points(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _process_loop(self):
        while self.running:
            # Get latest frame
            with self.lock:
                if self.latest_frame is None:
                    continue
                image_rgb = self.latest_frame.copy()
            
            # Flip BEFORE processing so landmarks match the mirror view
            image_rgb = cv2.flip(image_rgb, 1)

            # MediaPipe Tasks Processing
            # Create MP Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Ensure timestamp is strictly monotonically increasing
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms

            # Detect
            pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Prepare visualization image (BGR)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            # No flip here, already flipped!
            
            # Extract landmarks for logic
            pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
            pose_world_landmarks = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None
            
            right_hand_landmarks = None
            if hand_result.handedness:
                for i, handedness in enumerate(hand_result.handedness):
                    # Note: HandLandmarker uses "Left" and "Right" labels.
                    # Since we flipped the image, the "Right" hand in the image corresponds to the user's right hand?
                    # Wait, if we flip the image, the user sees a mirror.
                    # If I raise my right hand, in the mirror it's on the right side of the screen.
                    # MP usually detects it as "Right" if it looks like a right hand.
                    if handedness[0].category_name == "Right":
                        right_hand_landmarks = hand_result.hand_landmarks[i]
                        break

            # Draw skeleton with MORE COLOR
            if pose_landmarks:
                self._draw_landmarks(
                    image_bgr, 
                    pose_landmarks, 
                    self.POSE_CONNECTIONS,
                    landmark_color=(0, 255, 255), # Yellow joints
                    connection_color=(0, 255, 0), # Green connections
                    thickness=2,
                    radius=2
                )

            if right_hand_landmarks:
                self._draw_landmarks(
                    image_bgr, 
                    right_hand_landmarks, 
                    self.HAND_CONNECTIONS,
                    landmark_color=self.hand_landmark_color, # Custom
                    connection_color=self.hand_connection_color, # Custom
                    thickness=2,
                    radius=2
                )

            joints_raw = np.zeros(6)
            debug_info = []

            if pose_world_landmarks:
                lm = pose_world_landmarks
                def get_vec(idx): return np.array([lm[idx].x, lm[idx].y, lm[idx].z])
                
                r_shoulder = get_vec(PoseLandmark.RIGHT_SHOULDER)
                l_shoulder = get_vec(PoseLandmark.LEFT_SHOULDER)
                r_hip      = get_vec(PoseLandmark.RIGHT_HIP)
                l_hip      = get_vec(PoseLandmark.LEFT_HIP)
                r_elbow    = get_vec(PoseLandmark.RIGHT_ELBOW)
                r_wrist    = get_vec(PoseLandmark.RIGHT_WRIST)
                
                # --- Torso Frame ---
                shoulders_mid = (r_shoulder + l_shoulder) / 2
                hips_mid = (r_hip + l_hip) / 2
                torso_y = self._normalize(shoulders_mid - hips_mid)
                shoulders_vec = r_shoulder - l_shoulder
                torso_z = self._normalize(np.cross(shoulders_vec, torso_y))
                torso_x = self._normalize(np.cross(torso_y, torso_z))
                
                upper_arm = self._normalize(r_elbow - r_shoulder)
                comp_x = np.dot(upper_arm, torso_x)
                comp_y = np.dot(upper_arm, torso_y)
                comp_z = np.dot(upper_arm, torso_z)

                # Debug: show raw arm direction
                debug_info.append(f"Arm Y: {upper_arm[1]:.2f}")
                debug_info.append(f"comp_y: {comp_y:.2f}")

                # J1 Pan
                raw_pan = np.degrees(np.arctan2(comp_x, comp_z))
                joints_raw[0] = np.clip(-raw_pan, -90, 90)
                debug_info.append(f"J1 Pan: {-raw_pan:.1f}")

                # J2 Lift - MediaPipe Y points DOWN, so we negate comp_y
                # Arm up = negative comp_y (in MP coords) = positive lift
                raw_lift = np.degrees(np.arcsin(np.clip(-comp_y, -1.0, 1.0)))
                joints_raw[1] = np.clip(raw_lift, -90, 90)
                debug_info.append(f"J2 Lift: {raw_lift:.1f}")

                # J3 Elbow
                raw_elbow_angle = self._get_angle_from_points(r_shoulder, r_elbow, r_wrist)
                joints_raw[2] = self._map_range(raw_elbow_angle, 180, 40, 0, 90)
                joints_raw[2] = np.clip(joints_raw[2], 0, 90)
                debug_info.append(f"J3 Elbow: {joints_raw[2]:.1f}")

                # J4/J5 from IMU if available
                if self.imu_reader:
                    imu_roll, imu_flex = self.imu_reader.get_wrist_angles()
                    joints_raw[3] = imu_flex
                    joints_raw[4] = imu_roll
                    debug_info.append(f"J4 W-Flex(IMU): {joints_raw[3]:.1f}")
                    debug_info.append(f"J5 W-Roll(IMU): {joints_raw[4]:.1f}")
                else:
                    # J4 Flex (Locked)
                    joints_raw[3] = 0.0
                    debug_info.append(f"J4 W-Flex: {joints_raw[3]:.1f}")
                    
                    # J5 Roll (2D Heuristic)
                    if right_hand_landmarks and pose_landmarks:
                        # We need 2D landmarks for roll calculation
                        # right_hand_landmarks are normalized (0-1)
                        # pose_landmarks are normalized (0-1)
                        
                        hl_2d = right_hand_landmarks
                        h, w, _ = image_rgb.shape
                        
                        def get_p_2d(idx): return np.array([hl_2d[idx].x * w, hl_2d[idx].y * h])
                        
                        index_mcp_2d = get_p_2d(HandLandmark.INDEX_FINGER_MCP)
                        pinky_mcp_2d = get_p_2d(HandLandmark.PINKY_MCP)

                        # Forearm in 2D (from Pose)
                        # Use pose_landmarks (screen coords normalized)
                        lm_pose = pose_landmarks
                        f_start_2d = np.array([lm_pose[PoseLandmark.RIGHT_ELBOW].x * w, lm_pose[PoseLandmark.RIGHT_ELBOW].y * h])
                        f_end_2d = np.array([lm_pose[PoseLandmark.RIGHT_WRIST].x * w, lm_pose[PoseLandmark.RIGHT_WRIST].y * h])
                        forearm_vec_2d = self._normalize(f_end_2d - f_start_2d)

                        knuckle_vec_2d = self._normalize(pinky_mcp_2d - index_mcp_2d)
                        forearm_perp_2d = np.array([-forearm_vec_2d[1], forearm_vec_2d[0]])

                        raw_roll = np.degrees(np.arctan2(np.cross(forearm_perp_2d, knuckle_vec_2d), np.dot(forearm_perp_2d, knuckle_vec_2d)))
                        joints_raw[4] = np.clip(raw_roll, -90, 90)
                        debug_info.append(f"J5 W-Roll: {raw_roll:.1f}")

            # J6 Gripper
            if right_hand_landmarks:
                hl = right_hand_landmarks
                t_tip = np.array([hl[HandLandmark.THUMB_TIP].x, hl[HandLandmark.THUMB_TIP].y])
                i_tip = np.array([hl[HandLandmark.INDEX_FINGER_TIP].x, hl[HandLandmark.INDEX_FINGER_TIP].y])
                dist = np.linalg.norm(t_tip - i_tip)
                joints_raw[5] = self._map_range(dist, 0.02, 0.15, 0, 100)
                joints_raw[5] = np.clip(joints_raw[5], 0, 100)
                debug_info.append(f"J6 Gripper: {joints_raw[5]:.1f}")

            # Smoothing
            self.joints_smooth = (self.alpha * joints_raw) + ((1 - self.alpha) * self.joints_smooth)
            
            # Overlay Debug (right side)
            for i, text in enumerate(debug_info):
                cv2.putText(image_bgr, text, (self.width - 250, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow text

            # Draw IMU panel on left side (always, if IMU connected)
            if self.imu_reader:
                imu_roll, imu_flex = self.imu_reader.get_wrist_angles()
                raw_euler = self.imu_reader.get_raw_euler()

                imu_panel_x = 10
                imu_panel_y = 200  # Below any other overlays
                cv2.putText(image_bgr, "=== IMU DATA ===", (imu_panel_x, imu_panel_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)  # Magenta
                cv2.putText(image_bgr, f"Roll:  {imu_roll:+7.1f} deg", (imu_panel_x, imu_panel_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image_bgr, f"Flex:  {imu_flex:+7.1f} deg", (imu_panel_x, imu_panel_y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show raw euler
                if raw_euler is not None:
                    cv2.putText(image_bgr, f"Raw Euler (XYZ):", (imu_panel_x, imu_panel_y + 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    cv2.putText(image_bgr, f"  X: {raw_euler[0]:+7.1f}", (imu_panel_x, imu_panel_y + 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    cv2.putText(image_bgr, f"  Y: {raw_euler[1]:+7.1f}", (imu_panel_x, imu_panel_y + 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    cv2.putText(image_bgr, f"  Z: {raw_euler[2]:+7.1f}", (imu_panel_x, imu_panel_y + 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                else:
                    cv2.putText(image_bgr, "NO IMU DATA!", (imu_panel_x, imu_panel_y + 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red warning

            with self.lock:
                self.latest_result_img = image_bgr
                self.latest_joints = self.joints_smooth

    def get_action(self):
        # Non-blocking return of latest data
        with self.lock:
            if self.latest_result_img is None:
                return None, None
            return self.latest_joints.copy(), self.latest_result_img.copy()

if __name__ == "__main__":
    leader = WebcamLeader()
    leader.connect()
    try:
        while True:
            j, img = leader.get_action()
            if img is not None:
                cv2.imshow("Test", img)
            if cv2.waitKey(1) == 27: break
            time.sleep(0.001)
    finally:
        leader.disconnect()