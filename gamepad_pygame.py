#!/usr/bin/env python
"""Control SO101 arm with Xbox controller using pygame."""

import pygame
import time
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

ROBOT_PORT = "/dev/tty.usbmodem5AB90670321"
FPS = 30
STEP_SIZE = 2.0  # degrees per frame when stick is fully pressed
DEADZONE = 0.15


def main():
    # Initialize pygame
    pygame.init()
    pygame.joystick.init()

    # Check for joysticks
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No controller found!")
        return

    # Initialize first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Found controller: {joystick.get_name()}")
    print(f"  Axes: {joystick.get_numaxes()}")
    print(f"  Buttons: {joystick.get_numbuttons()}")

    # Connect to robot
    print("\nConnecting to robot...")
    config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="so101_follower",
        use_degrees=True
    )
    robot = SO101Follower(config)
    robot.connect()

    motor_names = list(robot.bus.motors.keys())
    print(f"Motors: {motor_names}")

    # Get initial positions
    obs = robot.get_observation()
    positions = {name: obs[f"{name}.pos"] for name in motor_names}

    print("\n" + "=" * 50)
    print("PYGAME XBOX CONTROLLER")
    print("=" * 50)
    print("Left stick   : shoulder_pan / shoulder_lift")
    print("Right stick  : wrist_roll / elbow_flex")
    print("LB/RB        : gripper close/open")
    print("Triggers     : wrist_flex")
    print("Ctrl+C       : Exit")
    print("=" * 50 + "\n")

    clock = pygame.time.Clock()

    try:
        running = True
        while running:
            # Process pygame events (required for joystick to work!)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Read axes
            # Common Xbox mapping:
            # Axis 0: Left stick X
            # Axis 1: Left stick Y
            # Axis 2: Right stick X (or left trigger on some)
            # Axis 3: Right stick Y (or right trigger on some)
            # Axis 4: Right stick X
            # Axis 5: Right stick Y (or triggers)

            num_axes = joystick.get_numaxes()

            left_x = joystick.get_axis(0) if num_axes > 0 else 0
            left_y = joystick.get_axis(1) if num_axes > 1 else 0

            # Try different axis mappings for right stick
            if num_axes >= 4:
                # Standard mapping
                right_x = joystick.get_axis(2) if num_axes > 2 else 0
                right_y = joystick.get_axis(3) if num_axes > 3 else 0
            else:
                right_x = 0
                right_y = 0

            # Triggers (might be axis 4 and 5, or 2 and 5)
            left_trigger = 0
            right_trigger = 0
            if num_axes > 4:
                left_trigger = (joystick.get_axis(4) + 1) / 2  # Convert -1..1 to 0..1
            if num_axes > 5:
                right_trigger = (joystick.get_axis(5) + 1) / 2

            # Apply deadzone
            if abs(left_x) < DEADZONE:
                left_x = 0
            if abs(left_y) < DEADZONE:
                left_y = 0
            if abs(right_x) < DEADZONE:
                right_x = 0
            if abs(right_y) < DEADZONE:
                right_y = 0

            # Read buttons
            num_buttons = joystick.get_numbuttons()
            lb = joystick.get_button(4) if num_buttons > 4 else 0  # Left bumper
            rb = joystick.get_button(5) if num_buttons > 5 else 0  # Right bumper

            # Update positions
            positions["shoulder_pan"] += -left_x * STEP_SIZE
            positions["shoulder_lift"] += -left_y * STEP_SIZE
            positions["elbow_flex"] += -right_y * STEP_SIZE
            positions["wrist_roll"] += right_x * STEP_SIZE

            # Wrist flex with triggers
            positions["wrist_flex"] += (right_trigger - left_trigger) * STEP_SIZE

            # Gripper with bumpers
            if rb:
                positions["gripper"] = min(100, positions["gripper"] + STEP_SIZE * 2)
            if lb:
                positions["gripper"] = max(0, positions["gripper"] - STEP_SIZE * 2)

            # Clamp positions
            for name in motor_names:
                if name == "gripper":
                    positions[name] = max(0, min(100, positions[name]))
                else:
                    positions[name] = max(-90, min(90, positions[name]))

            # Send to robot
            action = {f"{name}.pos": positions[name] for name in motor_names}
            robot.send_action(action)

            # Display
            print(f"\rL:({left_x:+.2f},{left_y:+.2f}) R:({right_x:+.2f},{right_y:+.2f}) | "
                  f"pan:{positions['shoulder_pan']:+5.1f} lift:{positions['shoulder_lift']:+5.1f} "
                  f"elbow:{positions['elbow_flex']:+5.1f} grip:{positions['gripper']:4.1f}  ",
                  end="", flush=True)

            clock.tick(FPS)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        robot.disconnect()
        pygame.quit()
        print("Done.")


if __name__ == "__main__":
    main()
