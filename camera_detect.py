"""
Fast Pose Detection with Arducam OV9782 @ 100fps + MediaPipe

Press 'q' to quit.
"""

import cv2
import time
import threading
import mediapipe as mp

# Camera settings - Arducam OV9782
CAMERA_INDEX = 0  # Arducam


class FastPoseDetector:
    def __init__(self):
        self.frame = None
        self.display_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.capture_fps = 0
        self.detect_fps = 0
        self.wrist_pos = None

        # Setup MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=0,  # 0=lite for speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def capture_loop(self, cap):
        """Capture frames as fast as possible"""
        prev_time = time.time()
        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                self.capture_fps = 10 / (now - prev_time)
                prev_time = now

            # Fix MJPG red tint by swapping R and B channels
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self.lock:
                self.frame = frame

    def detect_loop(self):
        """Run pose detection"""
        prev_time = time.time()
        frame_count = 0

        while self.running:
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None

            if frame is None:
                time.sleep(0.001)
                continue

            h, w = frame.shape[:2]

            # Run pose detection (frame is already RGB from capture)
            results = self.pose.process(frame)

            # Convert back to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                self.detect_fps = 10 / (now - prev_time)
                prev_time = now

            # Draw landmarks
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Get wrist positions
                landmarks = results.pose_landmarks.landmark
                lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                self.wrist_pos = {
                    'left': (int(lw.x * w), int(lw.y * h)),
                    'right': (int(rw.x * w), int(rw.y * h))
                }

            with self.lock:
                self.display_frame = frame

    def run(self):
        print(f"Opening Arducam OV9782 at index {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            print("Failed to open camera")
            return

        # Use MJPG for 100fps
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        cap.set(cv2.CAP_PROP_FPS, 100)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Verify settings
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

        print(f"Camera: {w}x{h} @ {fps}fps")
        print(f"Codec: {codec_str}")
        print("Press 'q' to quit\n")

        # Start threads
        cap_thread = threading.Thread(target=self.capture_loop, args=(cap,), daemon=True)
        det_thread = threading.Thread(target=self.detect_loop, daemon=True)
        cap_thread.start()
        det_thread.start()

        # Display loop
        while self.running:
            with self.lock:
                frame = self.display_frame.copy() if self.display_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            # Draw stats
            cv2.putText(frame, f"Capture: {int(self.capture_fps)} FPS", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detect: {int(self.detect_fps)} FPS", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.wrist_pos:
                lw = self.wrist_pos['left']
                rw = self.wrist_pos['right']
                cv2.putText(frame, f"L: {lw}  R: {rw}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"\rLeft: {lw} | Right: {rw}    ", end="", flush=True)

            cv2.imshow('Pose Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        print()
        self.pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FastPoseDetector()
    detector.run()
