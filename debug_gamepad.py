import hid
import time

# Find Xbox controller
devices = hid.enumerate()
for d in devices:
    if "Controller" in d["product_string"]:
        print(f"Found: {d['product_string']} - {d['manufacturer_string']}")
        device = hid.device()
        device.open_path(d["path"])
        device.set_nonblocking(1)
        
        print("\nMove sticks and press buttons. Ctrl+C to exit.\n")
        try:
            while True:
                data = device.read(64)
                if data:
                    # Print raw bytes
                    print(f"Bytes: {[hex(b) for b in data[:16]]}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            device.close()
        break
