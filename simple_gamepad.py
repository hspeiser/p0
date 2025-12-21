#!/usr/bin/env python
"""Direct joint control with PS5 DualSense controller using HID on macOS.

Controls (6 DOF):
  Left stick X  : shoulder_pan (side to side)
  Left stick Y  : shoulder_lift (up/down)
  Right stick Y : elbow_flex
  Right stick X : wrist_roll
  L1/R1         : wrist_flex (up/down)
  L2/R2         : gripper (close/open)

Press Ctrl+C to exit.
"""

import time

# Handle both hid package versions (hidapi vs hid)
try:
    import hid
    if hasattr(hid, 'device'):
        HID_DEVICE = hid.device
        HID_LEGACY = True
    else:
        HID_DEVICE = hid.Device
        HID_LEGACY = False
except ImportError:
    print("Install hid: pip install hidapi")
    exit(1)

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

# Configuration
ROBOT_PORT = "/dev/tty.usbmodem5AB90670321"
FPS = 60
STEP_SIZE = 0.6  # degrees per frame at full stick deflection
DEADZONE = 0.15

# PS5 DualSense
VID = 0x054c
PID = 0x0ce6


class PS5Controller:
    """PS5 DualSense controller reader using HID."""

    def __init__(self, deadzone=0.15):
        self.deadzone = deadzone
        self.device = None

        # Stick values (-1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Triggers (0.0 to 1.0)
        self.left_trigger = 0.0
        self.right_trigger = 0.0

        # Bumpers (L1/R1)
        self.l1 = False
        self.r1 = False

    def connect(self):
        if HID_LEGACY:
            self.device = HID_DEVICE()
            self.device.open(VID, PID)
            self.device.set_nonblocking(True)
            print(f"Connected to {self.device.get_manufacturer_string()} {self.device.get_product_string()}")
        else:
            self.device = HID_DEVICE(VID, PID)
            print(f"Connected to {self.device.manufacturer} {self.device.product}")

    def update(self):
        """Read latest data from controller."""
        if not self.device:
            return

        # Read latest data (non-blocking)
        try:
            if HID_LEGACY:
                # Drain buffer to get latest
                data = None
                for _ in range(50):
                    d = self.device.read(64)
                    if d:
                        data = d
                    else:
                        break
            else:
                data = self.device.read(64, timeout=0)
        except:
            return

        if not data or len(data) < 10:
            return

        # PS5 DualSense USB byte layout:
        # Byte 1: Left stick X (0-255, 128 center)
        # Byte 2: Left stick Y (0-255, 128 center)
        # Byte 3: Right stick X (0-255, 128 center)
        # Byte 4: Right stick Y (0-255, 128 center)
        # Byte 5: L2 trigger (0-255)
        # Byte 6: R2 trigger (0-255)

        self.left_x = (data[1] - 128) / 128.0
        self.left_y = (data[2] - 128) / 128.0
        self.right_x = (data[3] - 128) / 128.0
        self.right_y = (data[4] - 128) / 128.0

        self.left_trigger = data[5] / 255.0
        self.right_trigger = data[6] / 255.0

        # PS5 buttons are in byte 8
        # Bit 0: L1, Bit 1: R1, etc.
        if len(data) > 9:
            buttons = data[9]
            self.l1 = bool(buttons & 0x01)
            self.r1 = bool(buttons & 0x02)

        # Clamp to -1, 1
        self.left_x = max(-1, min(1, self.left_x))
        self.left_y = max(-1, min(1, self.left_y))
        self.right_x = max(-1, min(1, self.right_x))
        self.right_y = max(-1, min(1, self.right_y))

        # Apply deadzone
        if abs(self.left_x) < self.deadzone:
            self.left_x = 0
        if abs(self.left_y) < self.deadzone:
            self.left_y = 0
        if abs(self.right_x) < self.deadzone:
            self.right_x = 0
        if abs(self.right_y) < self.deadzone:
            self.right_y = 0

    def close(self):
        if self.device and HID_LEGACY:
            self.device.close()


def main():
    print("=" * 50)
    print("PS5 DUALSENSE - SO101 ARM")
    print("=" * 50)

    print("\nConnecting to PS5 controller...")
    controller = PS5Controller(deadzone=DEADZONE)
    controller.connect()

    print("\nConnecting to robot...")
    config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="so101_follower",
        use_degrees=True
    )
    robot = SO101Follower(config)
    robot.connect()

    motor_names = list(robot.bus.motors.keys())

    # Get initial positions
    obs = robot.get_observation()
    positions = {name: obs[f"{name}.pos"] for name in motor_names}

    print("\n" + "=" * 50)
    print("CONTROLS (6 DOF):")
    print("  Left stick X  : shoulder_pan")
    print("  Left stick Y  : shoulder_lift")
    print("  Right stick Y : elbow_flex")
    print("  Right stick X : wrist_roll")
    print("  L1/R1         : wrist_flex")
    print("  L2/R2         : gripper")
    print("  Ctrl+C        : Exit")
    print("=" * 50)
    print(f"Sensitivity: {STEP_SIZE} deg/frame | Deadzone: {DEADZONE}")
    print()

    try:
        while True:
            t0 = time.perf_counter()

            controller.update()

            # Map sticks to joints
            lx = controller.left_x
            ly = controller.left_y
            rx = controller.right_x
            ry = controller.right_y

            # Update joint positions (X axes inverted)
            positions["shoulder_pan"] += lx * STEP_SIZE
            positions["shoulder_lift"] += -ly * STEP_SIZE
            positions["elbow_flex"] += -ry * STEP_SIZE
            positions["wrist_roll"] += -rx * STEP_SIZE

            # Wrist flex control (L1/R1 bumpers)
            if controller.r1:
                positions["wrist_flex"] += STEP_SIZE
            elif controller.l1:
                positions["wrist_flex"] -= STEP_SIZE

            # Gripper control (triggers)
            if controller.right_trigger > 0.1:
                positions["gripper"] = min(100, positions["gripper"] + 2)
            elif controller.left_trigger > 0.1:
                positions["gripper"] = max(0, positions["gripper"] - 2)

            # Clamp positions to safe limits
            for name in motor_names:
                if name == "gripper":
                    positions[name] = max(0, min(100, positions[name]))
                else:
                    positions[name] = max(-90, min(90, positions[name]))

            # Send to robot
            action = {f"{name}.pos": positions[name] for name in motor_names}
            robot.send_action(action)

            # Display status
            l1r1 = f"{'L1' if controller.l1 else '  '}{'R1' if controller.r1 else '  '}"
            print(f"\rSticks:L({lx:+.2f},{ly:+.2f}) R({rx:+.2f},{ry:+.2f}) {l1r1} | "
                  f"pan:{positions['shoulder_pan']:+5.0f} lift:{positions['shoulder_lift']:+5.0f} "
                  f"elbow:{positions['elbow_flex']:+5.0f} wflex:{positions['wrist_flex']:+5.0f} "
                  f"roll:{positions['wrist_roll']:+5.0f} grip:{positions['gripper']:4.0f}  ",
                  end="", flush=True)

            # Maintain FPS
            elapsed = time.perf_counter() - t0
            if elapsed < 1 / FPS:
                time.sleep(1 / FPS - elapsed)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        controller.close()
        robot.disconnect()


if __name__ == "__main__":
    main()
