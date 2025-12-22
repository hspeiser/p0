# IMU + Webcam Teleop (Rust + Python)

This stack runs:
- Rust: IMU fusion, calibration, filtering, UDP command output.
- Python: MediaPipe hand tracking (UDP to Rust) + LeRobot IK and motor commands (UDP from Rust).

Default UDP ports:
- Camera -> Rust: 127.0.0.1:5005
- Robot state -> Rust: 127.0.0.1:5006
- Rust -> Robot commands: 127.0.0.1:5007

## Run order

1) Start camera sender (MediaPipe):

```
python examples/imu_webcam_to_so100/hand_udp_sender.py --mirror --hand any
```

2) Start robot bridge (IK + robot state):

```
python examples/imu_webcam_to_so100/robot_udp_bridge.py --port /dev/tty.usbmodemXXXX --speed-scale 0.05 --rerun
```

3) Build and run Rust fuser (calibrate on first run):

```
cargo run --release --manifest-path examples/imu_webcam_to_so100/rust/Cargo.toml -- \
  --imu-port /dev/tty.usbmodemYYYY \
  --calibrate \
  --speed-scale 0.05
```

On later runs, omit `--calibrate` and it will reuse `calibration.json`.

Calibration notes:
- The robot can stay locked. For Pose 1, align your hand to the robot tool’s current orientation and press ENTER.
- If calibration reports \"no camera samples\", run the camera sender with `--hand any` and keep your hand fully in view.
- The camera window overlays the hand skeleton and the thumb-index line used for the gripper pinch distance.

Controls:
- Press `x` in the Rust terminal to recenter the mapping to your current hand/robot pose.

## Notes
- The robot bridge uses the placeholder URDF `SO101/so101_new_calib.urdf`.
- The Rust fuser writes `calibration.json` in the current working directory.
- If MediaPipe models are missing, ensure `examples/webcam_to_so100_EE/models/hand_landmarker.task` exists.
- If IK fails, install the optional kinematics dependency (`placo`) for LeRobot.
