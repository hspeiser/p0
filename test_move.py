#!/usr/bin/env python
"""Simple test to move the arm."""

import time
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

ROBOT_PORT = "/dev/tty.usbmodem5AB90670321"

robot = SO101Follower(SO101FollowerConfig(port=ROBOT_PORT, id="so101_follower"))
robot.connect()

print("Connected! Reading current position...")
obs = robot.get_observation()
for k, v in obs.items():
    print(f"  {k}: {v:.2f}")

print("\nMoving shoulder_pan by 10 degrees in 3 seconds...")
time.sleep(3)

# Get current positions and move shoulder_pan slightly
current = {k.replace(".pos", ""): v for k, v in obs.items() if ".pos" in k}
current["shoulder_pan"] += 10

action = {f"{k}.pos": v for k, v in current.items()}
print(f"Sending: {action}")
robot.send_action(action)

print("Done! Disconnecting in 2 seconds...")
time.sleep(2)
robot.disconnect()
