#!/usr/bin/env python
"""Debug script to figure out Xbox controller byte mapping."""

import hid
import time

print("Looking for Xbox controller...")
devices = hid.enumerate()

xbox_device = None
for d in devices:
    product = d.get("product_string", "")
    if "Xbox" in product or "Controller" in product:
        print(f"Found: {product}")
        print(f"  Vendor ID: 0x{d['vendor_id']:04x}")
        print(f"  Product ID: 0x{d['product_id']:04x}")
        xbox_device = d
        break

if not xbox_device:
    print("No Xbox controller found!")
    exit(1)

device = hid.device()
device.open_path(xbox_device["path"])
device.set_nonblocking(True)

print("\n" + "="*60)
print("CONTROLLER BYTE MAPPING DEBUG")
print("="*60)
print("Move sticks and press buttons to see which bytes change.")
print("Press Ctrl+C to exit.\n")

last_data = None
try:
    while True:
        data = device.read(64)
        if data and data != last_data:
            last_data = data

            # Print raw bytes
            print(f"Length: {len(data)}")
            print("Bytes:", end=" ")
            for i, b in enumerate(data[:20]):
                print(f"[{i}]={b:3d}", end=" ")
            print()

            # Try to interpret as sticks (common layouts)
            if len(data) >= 14:
                # Xbox 360 style (signed 16-bit starting at byte 6)
                try:
                    import struct
                    lx = struct.unpack('<h', bytes(data[6:8]))[0]
                    ly = struct.unpack('<h', bytes(data[8:10]))[0]
                    rx = struct.unpack('<h', bytes(data[10:12]))[0]
                    ry = struct.unpack('<h', bytes(data[12:14]))[0]
                    print(f"  360 style: LX={lx:6d} LY={ly:6d} RX={rx:6d} RY={ry:6d}")
                except:
                    pass

            if len(data) >= 18:
                # Xbox One style (unsigned 16-bit starting at byte 10)
                lx = data[10] | (data[11] << 8)
                ly = data[12] | (data[13] << 8)
                rx = data[14] | (data[15] << 8)
                ry = data[16] | (data[17] << 8)
                print(f"  One style: LX={lx:5d} LY={ly:5d} RX={rx:5d} RY={ry:5d}")
                # Normalized
                lx_n = (lx - 32767) / 32767
                ly_n = (ly - 32767) / 32767
                rx_n = (rx - 32767) / 32767
                ry_n = (ry - 32767) / 32767
                print(f"  Normalized: LX={lx_n:+.2f} LY={ly_n:+.2f} RX={rx_n:+.2f} RY={ry_n:+.2f}")

            print()

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nDone!")
finally:
    device.close()
