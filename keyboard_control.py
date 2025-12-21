#!/usr/bin/env python
"""Control SO101 arm with keyboard."""

import sys
import tty
import termios
import time
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

ROBOT_PORT = "/dev/tty.usbmodem5AB90670321"
STEP_SIZE = 3.0  # degrees per keypress

def get_key():
    """Get a single keypress without waiting for enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle arrow keys (escape sequences)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == '[':
                if ch3 == 'A': return 'UP'
                if ch3 == 'B': return 'DOWN'
                if ch3 == 'C': return 'RIGHT'
                if ch3 == 'D': return 'LEFT'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    print("Connecting to robot...")
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

    print("\n" + "="*50)
    print("KEYBOARD CONTROL")
    print("="*50)
    print("W/S     : shoulder_lift up/down")
    print("A/D     : shoulder_pan left/right")
    print("UP/DOWN : elbow_flex")
    print("LEFT/RIGHT : wrist_roll")
    print("Q/E     : wrist_flex")
    print("O/C     : gripper open/close")
    print("ESC or X: Exit")
    print("="*50 + "\n")

    try:
        while True:
            key = get_key()

            if key in ['x', 'X', '\x1b', '\x03']:  # x, ESC, or Ctrl+C
                print("\nExiting...")
                break

            moved = False

            # Shoulder pan (A/D)
            if key in ['a', 'A']:
                positions["shoulder_pan"] -= STEP_SIZE
                moved = True
            elif key in ['d', 'D']:
                positions["shoulder_pan"] += STEP_SIZE
                moved = True

            # Shoulder lift (W/S)
            elif key in ['w', 'W']:
                positions["shoulder_lift"] += STEP_SIZE
                moved = True
            elif key in ['s', 'S']:
                positions["shoulder_lift"] -= STEP_SIZE
                moved = True

            # Elbow flex (UP/DOWN arrows)
            elif key == 'UP':
                positions["elbow_flex"] += STEP_SIZE
                moved = True
            elif key == 'DOWN':
                positions["elbow_flex"] -= STEP_SIZE
                moved = True

            # Wrist roll (LEFT/RIGHT arrows)
            elif key == 'LEFT':
                positions["wrist_roll"] -= STEP_SIZE
                moved = True
            elif key == 'RIGHT':
                positions["wrist_roll"] += STEP_SIZE
                moved = True

            # Wrist flex (Q/E)
            elif key in ['q', 'Q']:
                positions["wrist_flex"] -= STEP_SIZE
                moved = True
            elif key in ['e', 'E']:
                positions["wrist_flex"] += STEP_SIZE
                moved = True

            # Gripper (O/C)
            elif key in ['o', 'O']:
                positions["gripper"] = min(100, positions["gripper"] + STEP_SIZE * 2)
                moved = True
            elif key in ['c', 'C']:
                positions["gripper"] = max(0, positions["gripper"] - STEP_SIZE * 2)
                moved = True

            if moved:
                # Clamp positions
                for name in motor_names:
                    if name == "gripper":
                        positions[name] = max(0, min(100, positions[name]))
                    else:
                        positions[name] = max(-90, min(90, positions[name]))

                # Send action
                action = {f"{name}.pos": positions[name] for name in motor_names}
                robot.send_action(action)

                # Print current positions
                print(f"\rpan:{positions['shoulder_pan']:+6.1f} lift:{positions['shoulder_lift']:+6.1f} "
                      f"elbow:{positions['elbow_flex']:+6.1f} wrist:{positions['wrist_flex']:+6.1f} "
                      f"roll:{positions['wrist_roll']:+6.1f} grip:{positions['gripper']:5.1f}  ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        robot.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()
