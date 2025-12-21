#!/usr/bin/env python
"""Debug Xbox One controller via HID."""

import hid
import time

VID = 0x045e  # Microsoft
PID = 0x02ea  # Xbox One Controller

print(f"Opening Xbox One controller (VID: 0x{VID:04x}, PID: 0x{PID:04x})...")

device = hid.device()
try:
    device.open(VID, PID)
except Exception as e:
    print(f"Failed to open by VID/PID: {e}")
    print("\nTrying by path...")
    for d in hid.enumerate(VID, PID):
        print(f"  Trying path: {d['path']}")
        try:
            device.open_path(d['path'])
            print("  Success!")
            break
        except Exception as e2:
            print(f"  Failed: {e2}")

device.set_nonblocking(True)

print("\nMove sticks and press buttons. Ctrl+C to exit.\n")

try:
    while True:
        data = device.read(64)
        if data:
            print(f"Len={len(data):2d}: ", end="")
            for i, b in enumerate(data[:20]):
                print(f"{b:3d} ", end="")
            print()
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nDone!")
finally:
    device.close()
