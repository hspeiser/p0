#!/usr/bin/env python
"""Find which bytes correspond to the sticks."""

import time
import hid

VID = 0x045e
PID = 0x02ea

device = hid.device()
device.open(VID, PID)
device.set_nonblocking(True)

print("=" * 70)
print("FINDING STICK BYTES")
print("=" * 70)
print("\nWiggle sticks briefly then LET GO and hold still...")
print("Waiting for data...\n")

# Wait for any data
while True:
    data = device.read(64)
    if data:
        break
    time.sleep(0.01)

print("Got data! Now DON'T TOUCH for 2 seconds to get baseline...")
time.sleep(2)

# Get baseline (keep trying until we get stable readings)
baseline = None
for _ in range(100):
    data = device.read(64)
    if data:
        baseline = list(data)
    time.sleep(0.01)

if not baseline:
    print("No data received!")
    exit(1)

print(f"\nBaseline captured ({len(baseline)} bytes):")
for i in range(0, min(len(baseline), 20)):
    print(f"  byte[{i:2d}] = {baseline[i]:3d}")

print("\n" + "=" * 70)
print("NOW SLOWLY MOVE LEFT STICK LEFT-RIGHT ONLY")
print("Watch which byte indices change...")
print("=" * 70 + "\n")

try:
    while True:
        data = device.read(64)
        if data:
            # Find bytes that changed significantly from baseline
            changes = []
            for i, (b, base) in enumerate(zip(data, baseline)):
                diff = abs(b - base)
                if diff > 10:
                    changes.append(f"byte[{i}]={b}")

            if changes:
                print(f"Changed: {', '.join(changes)}")
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

device.close()
print("\n\nDone!")
