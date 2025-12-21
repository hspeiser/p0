#!/usr/bin/env python
"""Find the actual center/min/max values for your Xbox controller."""

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

print("=" * 60)
print("CONTROLLER CALIBRATION")
print("=" * 60)
print("\nDON'T TOUCH THE STICKS! Finding center values...")
print("Wait 3 seconds...\n")

time.sleep(1)

# Collect center values
centers = {"lx": [], "ly": [], "rx": [], "ry": []}

for i in range(60):
    data = device.read(64)
    if data and len(data) >= 18:
        lx_raw = struct.unpack('<H', bytes(data[10:12]))[0]
        ly_raw = struct.unpack('<H', bytes(data[12:14]))[0]
        rx_raw = struct.unpack('<H', bytes(data[14:16]))[0]
        ry_raw = struct.unpack('<H', bytes(data[16:18]))[0]

        centers["lx"].append(lx_raw)
        centers["ly"].append(ly_raw)
        centers["rx"].append(rx_raw)
        centers["ry"].append(ry_raw)

        print(f"\rRaw center: LX={lx_raw:5d} LY={ly_raw:5d} RX={rx_raw:5d} RY={ry_raw:5d}", end="", flush=True)
    time.sleep(0.05)

print("\n")

# Calculate average center
lx_center = sum(centers["lx"]) // len(centers["lx"]) if centers["lx"] else 32767
ly_center = sum(centers["ly"]) // len(centers["ly"]) if centers["ly"] else 32767
rx_center = sum(centers["rx"]) // len(centers["rx"]) if centers["rx"] else 32767
ry_center = sum(centers["ry"]) // len(centers["ry"]) if centers["ry"] else 32767

print(f"Center values found:")
print(f"  LX center: {lx_center}")
print(f"  LY center: {ly_center}")
print(f"  RX center: {rx_center}")
print(f"  RY center: {ry_center}")

print("\n" + "=" * 60)
print("Now move ALL sticks to their EXTREMES (full circles)")
print("Press Ctrl+C when done...")
print("=" * 60 + "\n")

mins = {"lx": 65535, "ly": 65535, "rx": 65535, "ry": 65535}
maxs = {"lx": 0, "ly": 0, "rx": 0, "ry": 0}

try:
    while True:
        data = device.read(64)
        if data and len(data) >= 18:
            lx_raw = struct.unpack('<H', bytes(data[10:12]))[0]
            ly_raw = struct.unpack('<H', bytes(data[12:14]))[0]
            rx_raw = struct.unpack('<H', bytes(data[14:16]))[0]
            ry_raw = struct.unpack('<H', bytes(data[16:18]))[0]

            mins["lx"] = min(mins["lx"], lx_raw)
            mins["ly"] = min(mins["ly"], ly_raw)
            mins["rx"] = min(mins["rx"], rx_raw)
            mins["ry"] = min(mins["ry"], ry_raw)

            maxs["lx"] = max(maxs["lx"], lx_raw)
            maxs["ly"] = max(maxs["ly"], ly_raw)
            maxs["rx"] = max(maxs["rx"], rx_raw)
            maxs["ry"] = max(maxs["ry"], ry_raw)

            print(f"\rLX:[{mins['lx']:5d}-{maxs['lx']:5d}] LY:[{mins['ly']:5d}-{maxs['ly']:5d}] "
                  f"RX:[{mins['rx']:5d}-{maxs['rx']:5d}] RY:[{mins['ry']:5d}-{maxs['ry']:5d}]",
                  end="", flush=True)
        time.sleep(0.01)

except KeyboardInterrupt:
    pass

print("\n\n" + "=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)
print(f"\nLX: min={mins['lx']:5d}, center={lx_center:5d}, max={maxs['lx']:5d}")
print(f"LY: min={mins['ly']:5d}, center={ly_center:5d}, max={maxs['ly']:5d}")
print(f"RX: min={mins['rx']:5d}, center={rx_center:5d}, max={maxs['rx']:5d}")
print(f"RY: min={mins['ry']:5d}, center={ry_center:5d}, max={maxs['ry']:5d}")

print("\n\nCopy these values to simple_gamepad.py!")

device.close()
