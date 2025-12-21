#!/usr/bin/env python
"""Show RAW values - don't touch sticks."""

import time
import struct
import hid

VID = 0x045e
PID = 0x02ea

device = hid.device()
device.open(VID, PID)
device.set_nonblocking(True)

print("DON'T TOUCH STICKS - showing raw values\n")

try:
    while True:
        data = device.read(64)
        if data and len(data) >= 18:
            lx = struct.unpack('<H', bytes(data[10:12]))[0]
            ly = struct.unpack('<H', bytes(data[12:14]))[0]
            rx = struct.unpack('<H', bytes(data[14:16]))[0]
            ry = struct.unpack('<H', bytes(data[16:18]))[0]

            # Show raw and normalized
            lx_n = (lx - 32767) / 32767.0
            ly_n = (ly - 32767) / 32767.0
            rx_n = (rx - 32767) / 32767.0
            ry_n = (ry - 32767) / 32767.0

            print(f"\rRAW: LX={lx:5d} LY={ly:5d} RX={rx:5d} RY={ry:5d} | NORM: LX={lx_n:+.2f} LY={ly_n:+.2f} RX={rx_n:+.2f} RY={ry_n:+.2f}", end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nDone!")
finally:
    device.close()
