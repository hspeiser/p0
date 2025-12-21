#!/usr/bin/env python
"""Reprogram a single Feetech motor ID.

Connect ONLY the motor you want to reprogram to the bus!
"""

from lerobot.motors.feetech import FeetechMotorsBus

PORT = "/dev/tty.usbmodem5AB90670321"

# SO101 Motor IDs:
# 1 = shoulder_pan
# 2 = shoulder_lift
# 3 = elbow_flex
# 4 = wrist_flex
# 5 = wrist_roll
# 6 = gripper

NEW_ID = 2  # <-- Change this to the ID you want

print(f"This will set the connected motor to ID {NEW_ID}")
print("Make sure ONLY ONE motor is connected!")
input("Press Enter to continue...")

bus = FeetechMotorsBus(port=PORT, motors={})
bus.connect()

# Scan for any motor (factory default is usually ID 1)
print("\nScanning for motors...")
for test_id in range(1, 10):
    try:
        pos = bus.read_with_motor_ids(bus.motor_model, test_id, "Present_Position")
        print(f"  Found motor at ID {test_id}")

        if test_id != NEW_ID:
            print(f"  Changing ID from {test_id} to {NEW_ID}...")
            bus.write_with_motor_ids(bus.motor_model, test_id, "ID", NEW_ID)
            print(f"  Done! Motor is now ID {NEW_ID}")
        else:
            print(f"  Motor already has ID {NEW_ID}")
        break
    except:
        pass

bus.disconnect()
print("\nDone!")
