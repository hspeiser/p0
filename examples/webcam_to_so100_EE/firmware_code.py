import time
import board
import busio
import json
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR

# Initialize I2C
i2c = None

# Try the singleton STEMMA_I2C first (best for built-in ports)
try:
    i2c = board.STEMMA_I2C()
    print("Using board.STEMMA_I2C()")
except Exception:
    pass

# Fallback to manual init if singleton failed
if i2c is None:
    try:
        # Pimoroni often uses SCL/SDA aliases for the connector
        # Note: Order is (SCL, SDA) for board.I2C(), but (SCL, SDA) for busio.I2C?
        # busio.I2C(scl, sda)
        i2c = busio.I2C(board.SCL, board.SDA)
        print("Using busio.I2C(SCL, SDA)")
    except Exception as e:
        print(f"I2C Init Error: {e}")
        # Last ditch: bitbangio? No, let's hope one of above works.
        while True: pass

try:
    bno = BNO08X_I2C(i2c)
    bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
    print("BNO085 Started.")
    
    while True:
        quat = bno.game_quaternion
        if quat:
            print(json.dumps({"q": [quat[3], quat[0], quat[1], quat[2]]}))
        time.sleep(0.01) 
except Exception as e:
    print(f"BNO Error: {e}")
    time.sleep(1)
