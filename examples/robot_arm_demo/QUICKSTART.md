# Quick Start Guide

## Running the Robot Arm Demo

### Option 1: Using pixi (Recommended)

```bash
cd /Users/leonardspeiser/Projects/rerun
pixi run -e py python examples/python/robot_arm/main.py --steps 100
```

### Option 2: Direct Python (if rerun-sdk is installed)

```bash
cd /Users/leonardspeiser/Projects/rerun/examples/python/robot_arm
python main.py --steps 100
```

### Option 3: With Le Robot mode

```bash
pixi run -e py python examples/python/robot_arm/main.py --lerobot-mode --steps 200
```

## What You'll See

- A 2-DOF robot arm with:
  - Blue base at the origin
  - Red first link (0.5m)
  - Orange first joint
  - Green second link (0.4m)
  - Yellow end effector
  - Orange trajectory line showing end effector path

- The arm will animate through a smooth figure-8 pattern

## Integration with Le Robot

See `lerobot_integration_example.py` for a complete example.

Basic pattern:
```python
from examples.python.robot_arm.main import log_robot_arm

# In your Le Robot loop:
joint_angles = obs['joint_positions']
theta1, theta2 = joint_angles[0], joint_angles[1]
log_robot_arm(theta1, theta2)
```

## Troubleshooting

If you get import errors:
1. Make sure you're in the rerun project directory
2. Install rerun-sdk: `pip install rerun-sdk` or use pixi
3. Check that numpy is installed: `pip install numpy`

If the viewer doesn't open:
- The script uses `rr.script_setup()` which handles viewer spawning
- You can also manually spawn: `rr.init("rerun_example_robot_arm", spawn=True)`


