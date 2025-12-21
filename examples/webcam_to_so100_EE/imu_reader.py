import serial
import json
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation

class IMUReader:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.latest_quat = None # [w, x, y, z]
        self.latest_euler = None # [roll, pitch, yaw] in degrees
        self.lock = threading.Lock()
        
        # Zeroing/Calibration offset
        self.offset_quat_inv = Rotation.identity() 

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"IMU connected on {self.port}")

            # Wait for valid data then zero
            print("Calibrating IMU (Hold arm neutral)...")
            # Wait until we actually have valid quaternion data
            max_wait = 5.0  # seconds
            start = time.time()
            while time.time() - start < max_wait:
                with self.lock:
                    if self.latest_quat is not None:
                        # Check if quaternion is valid (not all zeros)
                        q = self.latest_quat
                        norm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
                        if norm > 0.9:  # Valid quaternion has norm ~1
                            break
                time.sleep(0.1)

            self.zero()

        except Exception as e:
            print(f"Failed to connect to IMU: {e}")

    def disconnect(self):
        self.running = False
        time.sleep(0.2)  # Give read loop time to exit
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        print("IMU disconnected.")

    def zero(self):
        """Set current orientation as the 'zero' (neutral) position."""
        with self.lock:
            if self.latest_quat is not None:
                # Store inverse of current rotation to apply as offset
                current_rot = Rotation.from_quat([
                    self.latest_quat[1], # x
                    self.latest_quat[2], # y
                    self.latest_quat[3], # z
                    self.latest_quat[0]  # w
                ])
                self.offset_quat_inv = current_rot.inv()
                print("IMU Zeroed.")

    def _read_loop(self):
        lines_received = 0
        while self.running:
            try:
                if not self.ser or not self.ser.is_open:
                    break

                line = self.ser.readline().decode('utf-8').strip()
                if not line:
                    continue

                # Parse JSON {"q": [w, x, y, z]}
                try:
                    data = json.loads(line)
                    if "q" in data:
                        q_raw = data["q"]  # [w, x, y, z]

                        # Skip invalid quaternions (all zeros or bad norm)
                        norm = np.sqrt(q_raw[0]**2 + q_raw[1]**2 + q_raw[2]**2 + q_raw[3]**2)
                        if norm < 0.9:
                            continue

                        lines_received += 1

                        with self.lock:
                            self.latest_quat = q_raw

                            # Apply zero offset
                            # Current Raw Rotation
                            r_raw = Rotation.from_quat([q_raw[1], q_raw[2], q_raw[3], q_raw[0]])  # Scipy expects x,y,z,w

                            # Calibrated Rotation = Offset * Raw
                            r_cal = self.offset_quat_inv * r_raw

                            # Convert to Euler (Roll, Pitch, Yaw)
                            # Order depends on how BNO is mounted vs Arm.
                            # Assuming BNO flat on wrist:
                            # Roll (X) = Wrist Rotation (Supination/Pronation)
                            # Pitch (Y) = Wrist Flexion (Up/Down)
                            self.latest_euler = r_cal.as_euler('xyz', degrees=True)

                        # Debug: print every 100 samples
                        if lines_received % 100 == 0:
                            print(f"[IMU] Received {lines_received} samples, euler: {self.latest_euler}")

                except json.JSONDecodeError:
                    pass  # Ignore malformed lines

            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    print(f"[IMU] Read error: {e}")
                break

    def get_wrist_angles(self):
        """Returns (roll, flex) in degrees relative to zero."""
        with self.lock:
            if self.latest_euler is None:
                return 0.0, 0.0

            # Mapping depends on mounting orientation!
            # Assuming standard flat mount:
            # X = Roll (J5)
            # Y = Pitch (J4 - Flex)
            # You might need to swap/invert these based on physical mounting.

            roll = self.latest_euler[0]
            flex = self.latest_euler[1]

            # Clamp to robot limits
            roll = np.clip(roll, -90, 90)
            flex = np.clip(flex, -90, 90)

            return roll, flex

    def get_raw_euler(self):
        """Returns raw (roll, pitch, yaw) euler angles in degrees, or None if not available."""
        with self.lock:
            if self.latest_euler is None:
                return None
            return self.latest_euler.copy()

    def get_raw_quat(self):
        """Returns raw quaternion [w, x, y, z] or None if not available."""
        with self.lock:
            if self.latest_quat is None:
                return None
            return self.latest_quat.copy()

if __name__ == "__main__":
    # Test script
    # Find port first! ls /dev/tty.usbmodem*
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem12345" # UPDATE THIS
    
    imu = IMUReader(port)
    imu.connect()
    try:
        while True:
            roll, flex = imu.get_wrist_angles()
            print(f"\rWrist Roll: {roll:.1f}  Flex: {flex:.1f}   ", end="")
            time.sleep(0.05)
    except KeyboardInterrupt:
        imu.disconnect()
