<!--[metadata]
title = "Simple Robot Arm Simulation"
tags = ["3D", "Robotics", "Kinematics", "Le Robot"]
-->

# Simple Robot Arm Simulation

A simple 2-DOF (degree of freedom) robot arm simulation that demonstrates forward kinematics and can be easily integrated with the Le Robot simulator.

## Features

- **Forward Kinematics**: Computes end-effector position from joint angles
- **Visualization**: Shows robot links, joints, and end-effector trajectory
- **Animation**: Smooth movement along a predefined trajectory
- **Le Robot Ready**: Designed for easy integration with Le Robot simulator

## Usage

### Prerequisites

Install dependencies:
```bash
pip install rerun-sdk numpy
```

### Basic Usage

**Option 1: Using pixi (if installed)**
```bash
pixi run -e py python examples/python/robot_arm/main.py
```

**Option 2: Direct Python (recommended if pixi not available)**
```bash
cd examples/python/robot_arm
python main.py
```

**Option 3: Using the run script**
```bash
cd examples/python/robot_arm
./run.sh
```

### With Custom Steps

```bash
# With pixi
pixi run -e py python examples/python/robot_arm/main.py --steps 1000

# Direct Python
python examples/python/robot_arm/main.py --steps 1000
```

### Le Robot Integration Mode

```bash
python examples/python/robot_arm/main.py --lerobot-mode --steps 200
```

## Integration with Le Robot

### Quick Integration

See `lerobot_integration_example.py` for a complete example of how to integrate with Le Robot.

### Basic Integration Pattern

Add this to your Le Robot training/evaluation script:

```python
import rerun as rr
from examples.python.robot_arm.main import forward_kinematics, log_robot_arm

# Initialize Rerun (once at start)
rr.init("lerobot_training", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

# In your training loop:
for episode in range(num_episodes):
    obs = env.reset()

    for step in range(episode_length):
        # Get joint angles from Le Robot observation
        joint_angles = obs['joint_positions']  # Adjust based on your setup
        theta1, theta2 = joint_angles[0], joint_angles[1]

        # Log to Rerun
        rr.set_time_sequence("episode", episode)
        rr.set_time_sequence("step", step)

        # Visualize robot arm
        log_robot_arm(theta1, theta2)

        # Continue with your Le Robot logic
        action = policy(obs)
        obs, reward, done, info = env.step(action)

        if done:
            break
```

### Testing the Integration

Run the integration example:

```bash
pixi run -e py python examples/python/robot_arm/lerobot_integration_example.py
```

## Robot Arm Structure

The robot arm consists of:
- **Base**: Fixed at origin (0, 0, 0.1)
- **Link 1**: Length 0.5m, rotates around base (θ₁)
- **Link 2**: Length 0.4m, rotates around joint 1 (θ₂)
- **End Effector**: Located at the tip of Link 2

## Customization

You can easily modify the robot parameters at the top of `main.py`:

```python
BASE_HEIGHT = 0.1      # Height of the base
LINK1_LENGTH = 0.5     # Length of first link
LINK2_LENGTH = 0.4     # Length of second link
JOINT_RADIUS = 0.03    # Visual size of joints
LINK_RADIUS = 0.02     # Visual size of links
```

## Extending to More DOF

To add more degrees of freedom, extend the `forward_kinematics` function:

```python
def forward_kinematics_3dof(theta1: float, theta2: float, theta3: float):
    # Add third joint computation
    ...
```


