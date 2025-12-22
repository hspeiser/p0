#!/usr/bin/env python
import argparse
import json
import socket
import time
import numpy as np

from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.rotation import Rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Robot serial port")
    parser.add_argument("--urdf", default="SO101/so101_new_calib.urdf")
    parser.add_argument("--cmd-listen", default="127.0.0.1:5007")
    parser.add_argument("--state-send", default="127.0.0.1:5006")
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--speed-scale", type=float, default=0.05)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-relative-target", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    host_cmd, port_cmd = args.cmd_listen.split(":")
    cmd_addr = (host_cmd, int(port_cmd))

    host_state, port_state = args.state_send.split(":")
    state_addr = (host_state, int(port_state))

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.bind(cmd_addr)
    cmd_sock.setblocking(False)

    state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    config = SO100FollowerConfig(
        port=args.port,
        id="imu_webcam_teleop",
        use_degrees=True,
        max_relative_target=args.max_relative_target,
    )
    follower = SO100Follower(config)
    follower.connect()

    kinematics = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    motor_names = list(follower.bus.motors.keys())
    motor_indices = {name: i for i, name in enumerate(motor_names)}
    speed_scale = max(0.01, min(args.speed_scale, 1.0))
    q_curr = None
    last_cmd = None
    last_cmd_t = 0.0

    dt = 1.0 / max(args.rate, 1.0)

    if args.rerun:
        init_rerun(session_name="imu_webcam_teleop")

    print("Robot bridge running. Press Ctrl+C to stop.")
    try:
        while True:
            t0 = time.time()

            obs = follower.get_observation()
            q_raw = np.array([float(obs[f"{n}.pos"]) for n in motor_names], dtype=float)
            t_fk = kinematics.forward_kinematics(q_raw)
            ee_pos = t_fk[:3, 3]
            ee_rot = Rotation.from_matrix(t_fk[:3, :3])
            q_xyzw = ee_rot.as_quat()
            ee_quat = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

            state_msg = {
                "kind": "robot_state",
                "t": time.time(),
                "ee_pos": [float(v) for v in ee_pos],
                "ee_quat": ee_quat,
            }
            state_sock.sendto(json.dumps(state_msg, separators=(",", ":")).encode("utf-8"), state_addr)

            try:
                data, _ = cmd_sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                if msg.get("kind") == "cmd":
                    last_cmd = msg
                    last_cmd_t = time.time()
            except BlockingIOError:
                pass

            if last_cmd is not None and (time.time() - last_cmd_t) < 0.2:
                ee_pos_cmd = last_cmd.get("ee_pos", ee_pos.tolist())
                ee_rotvec = last_cmd.get("ee_rotvec", [0.0, 0.0, 0.0])
                gripper = float(last_cmd.get("gripper", 0.0))

                t_des = np.eye(4, dtype=float)
                t_des[:3, :3] = Rotation.from_rotvec(np.array(ee_rotvec, dtype=float)).as_matrix()
                t_des[:3, 3] = np.array(ee_pos_cmd, dtype=float)

                if q_curr is None:
                    q_curr = q_raw.copy()

                q_target = kinematics.inverse_kinematics(
                    q_curr,
                    t_des,
                    position_weight=0.05,
                    orientation_weight=1.0,
                )
                q_cmd = q_raw + speed_scale * (q_target - q_raw)
                q_curr = q_cmd

                action = {}
                for i, name in enumerate(motor_names):
                    if name == "gripper":
                        action["gripper.pos"] = float(np.clip(gripper, 0.0, 100.0))
                    else:
                        action[f"{name}.pos"] = float(q_cmd[i])

                follower.send_action(action)

                if args.rerun:
                    obs_log = {
                        "ee.x": float(ee_pos[0]),
                        "ee.y": float(ee_pos[1]),
                        "ee.z": float(ee_pos[2]),
                        "wrist_flex.pos": float(q_raw[motor_indices["wrist_flex"]]),
                        "wrist_roll.pos": float(q_raw[motor_indices["wrist_roll"]]),
                        "gripper.pos": float(obs["gripper.pos"]),
                    }
                    action_log = {
                        "ee.x": float(ee_pos_cmd[0]),
                        "ee.y": float(ee_pos_cmd[1]),
                        "ee.z": float(ee_pos_cmd[2]),
                        "wrist_flex.pos": float(q_cmd[motor_indices["wrist_flex"]]),
                        "wrist_roll.pos": float(q_cmd[motor_indices["wrist_roll"]]),
                        "gripper.pos": float(gripper),
                    }
                    log_rerun_data(observation=obs_log, action=action_log)

            elapsed = time.time() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
