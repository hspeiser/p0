#!/usr/bin/env python
"""Test controller input only - no robot."""

import time
import struct
import hid

VID = 0x045e
PID = 0x02ea

print("Opening controller...")
device = hid.device()
device.open(VID, PID)
device.set_nonblocking(True)
print("Connected!\n")

print("Move sticks and watch the values. They should respond instantly.")
print("If values drift or lag, that's the problem.\n")
print("Ctrl+C to exit\n")

DEADZONE = 0.2

try:
    while True:
        # Read ONE packet
        data = device.read(64)

        if data and len(data) >= 18:
            lx_raw = struct.unpack('<H', bytes(data[10:12]))[0]
            ly_raw = struct.unpack('<H', bytes(data[12:14]))[0]
            rx_raw = struct.unpack('<H', bytes(data[14:16]))[0]
            ry_raw = struct.unpack('<H', bytes(data[16:18]))[0]

            lx = max(-1, min(1, (lx_raw - 32767) / 32767.0))
            ly = max(-1, min(1, (ly_raw - 32767) / 32767.0))
            rx = max(-1, min(1, (rx_raw - 32767) / 32767.0))
            ry = max(-1, min(1, (ry_raw - 32767) / 32767.0))

            # Apply deadzone
            if abs(lx) < DEADZONE: lx = 0
            if abs(ly) < DEADZONE: ly = 0
            if abs(rx) < DEADZONE: rx = 0
            if abs(ry) < DEADZONE: ry = 0

            print(f"\rLX:{lx:+.2f} LY:{ly:+.2f} RX:{rx:+.2f} RY:{ry:+.2f}   ", end="", flush=True)

        time.sleep(0.016)  # ~60fps

except KeyboardInterrupt:
    print("\nDone!")
finally:
    device.close()
