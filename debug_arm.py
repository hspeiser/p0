#!/usr/bin/env python
"""Debug script to diagnose SO101 arm issues."""

import time
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

ROBOT_PORT = "/dev/tty.usbmodem5AB90670321"

print("=" * 50)
print("SO101 ARM DIAGNOSTIC")
print("=" * 50)

# Create robot
print("\n1. Creating robot instance...")
robot = SO101Follower(SO101FollowerConfig(port=ROBOT_PORT, id="so101_follower"))

# Connect
print("\n2. Connecting to robot...")
try:
    robot.connect()
    print("   ✓ Connected successfully!")
except Exception as e:
    print(f"   ✗ Connection failed: {e}")
    exit(1)

# Check torque status
print("\n3. Checking torque status...")
try:
    torque_status = robot.bus.sync_read("Torque_Enable")
    print(f"   Torque status: {torque_status}")
    all_enabled = all(v == 1 for v in torque_status.values())
    print(f"   All motors torque enabled: {all_enabled}")
except Exception as e:
    print(f"   ✗ Failed to read torque: {e}")

# Read current positions
print("\n4. Reading current motor positions...")
try:
    obs = robot.get_observation()
    print("   Current positions:")
    for k, v in obs.items():
        if ".pos" in k:
            print(f"     {k}: {v:.2f}")
except Exception as e:
    print(f"   ✗ Failed to read positions: {e}")

# Try to enable torque explicitly
print("\n5. Explicitly enabling torque...")
try:
    robot.bus.enable_torque()
    print("   ✓ Torque enabled!")
except Exception as e:
    print(f"   ✗ Failed to enable torque: {e}")

# Test small movement
print("\n6. Testing movement (shoulder_pan +5)...")
print("   Will move in 2 seconds...")
time.sleep(2)

try:
    obs = robot.get_observation()
    current = {k.replace(".pos", ""): v for k, v in obs.items() if ".pos" in k}

    # Small movement
    target = current.copy()
    target["shoulder_pan"] += 5

    action = {f"{k}.pos": v for k, v in target.items()}
    print(f"   Sending action: shoulder_pan = {target['shoulder_pan']:.2f}")
    robot.send_action(action)
    print("   ✓ Action sent!")

    time.sleep(1)

    # Read new position
    new_obs = robot.get_observation()
    new_pos = new_obs["shoulder_pan.pos"]
    print(f"   New position: {new_pos:.2f}")

    if abs(new_pos - target["shoulder_pan"]) < 2:
        print("   ✓ ARM MOVED SUCCESSFULLY!")
    else:
        print("   ✗ Arm did not reach target position")
        print("   Possible issues:")
        print("     - Motor torque might be disabled by protection")
        print("     - Mechanical obstruction")
        print("     - Motor calibration mismatch")

except Exception as e:
    print(f"   ✗ Movement test failed: {e}")

# Disconnect
print("\n7. Disconnecting...")
robot.disconnect()
print("   ✓ Done!")
