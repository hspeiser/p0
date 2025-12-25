#!/usr/bin/env python3
"""
Wire the Le Robot arm URDF to Rerun with a simple inverse-kinematics loop.

What it does:
  - Loads your URDF (with package://assets resolved)
  - Builds a kinematic chain from base -> jaw
  - Runs a crude CCD-style IK to reach a sequence of goal positions
  - Logs joint angles to Rerun so you can see the arm move

Usage (inside your venv):
  RERUN_URDF_PACKAGE_PATH=/Users/leonardspeiser/Desktop/Archive \
  python examples/python/robot_arm/lerobot_control.py \
    --urdf /Users/leonardspeiser/Desktop/Archive/robot.urdf \
    --goals "0.25 0.05 0.15" "0.10 0.10 0.20"
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rerun as rr
from rerun.archetypes import Transform3D


# ----------------- URDF parsing (reuses logic from load_external_urdf) -----------------

@dataclass
class JointSpec:
    name: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


def parse_joints(urdf_path: Path) -> list[JointSpec]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints: list[JointSpec] = []
    for joint in root.findall("joint"):
        jtype = joint.get("type", "fixed")
        if jtype not in ("revolute", "continuous"):
            continue
        name = joint.get("name", "")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link", "")
        child = child_el.get("link", "")

        origin_el = joint.find("origin")
        if origin_el is not None:
            xyz_list = [float(x) for x in origin_el.get("xyz", "0 0 0").split()][:3]
            xyz_list += [0.0] * (3 - len(xyz_list))
            rpy_list = [float(r) for r in origin_el.get("rpy", "0 0 0").split()][:3]
            rpy_list += [0.0] * (3 - len(rpy_list))
            origin_xyz = (xyz_list[0], xyz_list[1], xyz_list[2])
            origin_rpy = (rpy_list[0], rpy_list[1], rpy_list[2])
        else:
            origin_xyz = (0.0, 0.0, 0.0)
            origin_rpy = (0.0, 0.0, 0.0)
        axis_el = joint.find("axis")
        if axis_el is not None:
            axis_list = [float(a) for a in axis_el.get("xyz", "0 0 1").split()][:3]
            axis_list += [0.0] * (3 - len(axis_list))
            axis = (axis_list[0], axis_list[1], axis_list[2])
        else:
            axis = (0.0, 0.0, 1.0)

        joints.append(
            JointSpec(
                name=name,
                parent=parent,
                child=child,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
            )
        )
    return joints


def rewrite_urdf_assets(urdf_path: Path, package_path: Path) -> Path:
    text = urdf_path.read_text(encoding="utf-8")
    assets_abs = (package_path / "assets").resolve()
    prefix = "package://assets/"
    replacement = f"file://{assets_abs}/"
    if prefix in text:
        rewritten = text.replace(prefix, replacement)
        tmp_path = urdf_path.parent / f"{urdf_path.stem}_rewritten.urdf"
        tmp_path.write_text(rewritten, encoding="utf-8")
        return tmp_path
    return urdf_path


# ----------------- Kinematics helpers -----------------

def rpy_to_mat(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def axis_angle_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def axis_angle_to_quat(axis: tuple[float, float, float], angle: float) -> rr.Quaternion:
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0.0:
        return rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])
    ax /= norm
    ay /= norm
    az /= norm
    s = math.sin(angle * 0.5)
    c = math.cos(angle * 0.5)
    return rr.Quaternion(xyzw=[ax * s, ay * s, az * s, c])


def euler_to_quat(rpy: tuple[float, float, float]) -> rr.Quaternion:
    roll, pitch, yaw = rpy
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return rr.Quaternion(xyzw=[qx, qy, qz, qw])


def quat_mul(a: rr.Quaternion, b: rr.Quaternion) -> rr.Quaternion:
    ax, ay, az, aw = a.xyzw
    bx, by, bz, bw = b.xyzw
    return rr.Quaternion(
        xyzw=[
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def build_chain(joints: list[JointSpec], tip: str = "jaw", root: str = "base") -> list[JointSpec]:
    by_child = {j.child: j for j in joints}
    chain_rev: list[JointSpec] = []
    cur = tip
    while cur in by_child:
        j = by_child[cur]
        chain_rev.append(j)
        if j.parent == root:
            break
        cur = j.parent
    return list(reversed(chain_rev))


def forward_transforms(chain: list[JointSpec], angles: List[float]) -> list[np.ndarray]:
    T_list = []
    T = np.eye(4)
    for j, a in zip(chain, angles):
        # origin transform
        R0 = rpy_to_mat(j.origin_rpy)
        t0 = np.array(j.origin_xyz)
        T_origin = np.eye(4)
        T_origin[:3, :3] = R0
        T_origin[:3, 3] = t0
        # joint rotation
        Rj = axis_angle_rot(np.array(j.axis), a)
        T_rot = np.eye(4)
        T_rot[:3, :3] = Rj
        # compose
        T = T @ T_origin @ T_rot
        T_list.append(T.copy())
    return T_list


def ccd_ik(
    chain: list[JointSpec],
    angles: List[float],
    goal: np.ndarray,
    iters: int = 50,
    step: float = 0.2,
    damping: float = 0.6,
) -> List[float]:
    """
    CCD IK with damping to prevent overshoot.

    damping: 0.0-1.0, lower = more damping = smoother but slower convergence
    """
    angles = list(angles)
    prev_dist = float('inf')

    for iteration in range(iters):
        T_list = forward_transforms(chain, angles)
        eff_pos = T_list[-1][:3, 3]
        curr_dist = np.linalg.norm(eff_pos - goal)

        # Adaptive step: reduce step size as we get closer or if oscillating
        adaptive_step = step * min(1.0, curr_dist * 10)  # Scale down when close
        if curr_dist > prev_dist:  # Overshooting, reduce step
            adaptive_step *= 0.5

        for i in reversed(range(len(chain))):
            j = chain[i]
            joint_pos = T_list[i - 1][:3, 3] if i > 0 else np.zeros(3)
            axis_world = T_list[i][:3, :3] @ (np.array(j.axis) / (np.linalg.norm(j.axis) + 1e-9))
            v_eff = eff_pos - joint_pos
            v_goal = goal - joint_pos
            v_eff_proj = v_eff - axis_world * np.dot(axis_world, v_eff)
            v_goal_proj = v_goal - axis_world * np.dot(axis_world, v_goal)
            if np.linalg.norm(v_eff_proj) < 1e-6 or np.linalg.norm(v_goal_proj) < 1e-6:
                continue
            v_eff_p = v_eff_proj / np.linalg.norm(v_eff_proj)
            v_goal_p = v_goal_proj / np.linalg.norm(v_goal_proj)
            cross = np.cross(v_eff_p, v_goal_p)
            dot = np.clip(np.dot(v_eff_p, v_goal_p), -1.0, 1.0)
            ang = math.atan2(np.dot(cross, axis_world), dot)
            # Clamp and apply damping
            ang = max(min(ang, adaptive_step), -adaptive_step) * damping
            angles[i] += ang

            # Update transforms after each joint for smoother convergence
            T_list = forward_transforms(chain, angles)
            eff_pos = T_list[-1][:3, 3]

        prev_dist = curr_dist

        # Early exit if close enough
        if curr_dist < 1e-4:
            break

    return angles


def log_joint_angles(chain: list[JointSpec], angles: List[float], step: int) -> None:
    rr.set_time("step", sequence=step)
    for j, a in zip(chain, angles):
        # Compose origin rotation with joint rotation about the local axis
        rot_origin = euler_to_quat(j.origin_rpy)
        rot_dyn = axis_angle_to_quat(j.axis, a)
        rot = quat_mul(rot_origin, rot_dyn)
        rr.log(
            f"world/urdf/joints/{j.name}",
            Transform3D(
                translation=list(j.origin_xyz),
                rotation=rot,
                parent_frame=j.parent,
                child_frame=j.child,
            ),
        )
        # Plot joint angle (degrees)
        rr.log(f"plots/joints/{j.name}", rr.Scalars([math.degrees(a)]))


def log_status(step: int, goal: np.ndarray, eff: np.ndarray, dist: float, subgoal: np.ndarray) -> None:
    rr.set_time("step", sequence=step)
    rr.log(
        "world/status",
        rr.TextLog(
            f"Goal: ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}) | "
            f"StepGoal: ({subgoal[0]:.3f}, {subgoal[1]:.3f}, {subgoal[2]:.3f}) | "
            f"EE: ({eff[0]:.3f}, {eff[1]:.3f}, {eff[2]:.3f}) | "
            f"dist={dist*1000:.1f} mm"
        ),
    )
    # Plot EE position (cm) and distance (mm)
    rr.log("plots/ee/x", rr.Scalars([eff[0] * 100]))
    rr.log("plots/ee/y", rr.Scalars([eff[1] * 100]))
    rr.log("plots/ee/z", rr.Scalars([eff[2] * 100]))
    rr.log("plots/distance_mm", rr.Scalars([dist * 1000]))
    # Visualize goal and EE in 3D
    rr.log("world/goal", rr.Points3D([goal], colors=[[0, 255, 0]], radii=[0.01]))
    rr.log("world/ee", rr.Points3D([eff], colors=[[255, 255, 0]], radii=[0.008]))


def generate_random_goal(arm_reach: float = 0.25) -> np.ndarray:
    """Generate a random reachable goal within the arm's workspace."""
    # Random point in a sphere, biased toward reachable positions
    r = arm_reach * (0.3 + 0.7 * np.random.random())  # 30%-100% of reach
    theta = np.random.uniform(0, 2 * np.pi)  # Azimuth
    phi = np.random.uniform(0.2, np.pi * 0.6)  # Elevation (avoid straight up/down)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi) + 0.05  # Offset up from base

    return np.array([x, y, z])


# ----------------- Script entry -----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Le Robot IK control to Rerun.")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--package-path",
        type=Path,
        default=None,
        help="Folder containing assets/; defaults to urdf parent.",
    )
    parser.add_argument(
        "--goals",
        nargs="+",
        type=float,
        default=[0.15, 0.0, 0.02],  # pick-from-ground style goal ~90 deg down from start
        help="Goal positions x y z ... (multiple of 3 numbers).",
    )
    parser.add_argument("--sleep", type=float, default=8.0)
    parser.add_argument(
        "--pos-step",
        type=float,
        default=0.001,  # 0.1 cm
        help="Max linear progress toward goal per frame (meters).",
    )
    parser.add_argument(
        "--ik-step",
        type=float,
        default=0.05,
        help="Max joint angle change per CCD update (radians).",
    )
    parser.add_argument(
        "--ik-iters",
        type=int,
        default=8,
        help="CCD iterations per frame.",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.5,
        help="IK damping factor 0.0-1.0 (lower = smoother, less overshoot).",
    )
    parser.add_argument(
        "--goal-tol",
        type=float,
        default=1e-4,  # 0.1 mm
        help="Distance tolerance to consider goal reached (meters).",
    )
    parser.add_argument(
        "--max-frames-per-goal",
        type=int,
        default=1200,
        help="Safety cap on frames per goal; increase for tighter convergence.",
    )
    parser.add_argument(
        "--lock-gripper",
        action="store_true",
        help="Keep the last joint (gripper) fixed at 0 angle.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Continuously move to random reachable goals.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause duration (seconds) at each goal when using --random.",
    )
    parser.add_argument(
        "--num-goals",
        type=int,
        default=0,
        help="Number of random goals (0 = infinite loop).",
    )
    args = parser.parse_args()

    urdf = args.urdf.expanduser().resolve()
    package = (args.package_path or urdf.parent).expanduser().resolve()
    os.environ["RERUN_URDF_PACKAGE_PATH"] = str(package)

    rr.init("lerobot_ik", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    urdf_to_load = rewrite_urdf_assets(urdf, package)
    rr.log_file_from_path(urdf_to_load, static=True)

    joints_all = parse_joints(urdf_to_load)
    chain = build_chain(joints_all, tip="jaw", root="base")
    if not chain:
        print("Failed to build chain base->jaw.")
        return
    print("Chain order:", [j.name for j in chain])

    # initial angles zero
    angles = [0.0 for _ in chain]

    def move_to_goal(g: np.ndarray, step: int) -> Tuple[List[float], int]:
        """Move to a single goal, return updated angles and step count."""
        nonlocal angles
        frames = 0
        while True:
            T_list = forward_transforms(chain, angles)
            eff = T_list[-1][:3, 3]
            delta = g - eff
            dist = np.linalg.norm(delta)
            if dist < args.goal_tol:
                break
            # move a small linear step toward goal
            step_vec = delta if dist < args.pos_step else delta * (args.pos_step / dist)
            subgoal = eff + step_vec
            angles = ccd_ik(chain, angles, subgoal, iters=args.ik_iters, step=args.ik_step, damping=args.damping)
            if args.lock_gripper and len(angles) >= 1:
                angles[-1] = 0.0  # freeze gripper joint
            log_joint_angles(chain, angles, step)
            log_status(step, g, eff, dist, subgoal)
            step += 1
            frames += 1
            if frames >= args.max_frames_per_goal:
                break
        # refinement pass if still above tolerance
        T_list = forward_transforms(chain, angles)
        eff = T_list[-1][:3, 3]
        dist = np.linalg.norm(g - eff)
        if dist > args.goal_tol:
            refine_pos_step = args.pos_step * 0.5
            refine_ik_step = args.ik_step * 0.5
            refine_iters = max(4, args.ik_iters // 2)
            for _ in range(200):
                T_list = forward_transforms(chain, angles)
                eff = T_list[-1][:3, 3]
                delta = g - eff
                dist = np.linalg.norm(delta)
                if dist < args.goal_tol:
                    break
                step_vec = delta if dist < refine_pos_step else delta * (refine_pos_step / dist)
                subgoal = eff + step_vec
                angles = ccd_ik(chain, angles, subgoal, iters=refine_iters, step=refine_ik_step, damping=args.damping * 0.8)
                if args.lock_gripper and len(angles) >= 1:
                    angles[-1] = 0.0
                log_joint_angles(chain, angles, step)
                log_status(step, g, eff, dist, subgoal)
                step += 1
        return angles, step

    step = 0

    if args.random:
        # Random goal mode - continuously move to random reachable points
        goal_count = 0
        print("Random goal mode - press Ctrl+C to stop")
        try:
            while args.num_goals == 0 or goal_count < args.num_goals:
                g = generate_random_goal(arm_reach=0.28)
                goal_count += 1
                print(f"Goal {goal_count}: ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})")
                angles, step = move_to_goal(g, step)
                # Pause at goal
                print(f"  Reached! Pausing {args.pause}s...")
                # Log a few frames at rest to show pause
                for _ in range(int(args.pause * 30)):
                    T_list = forward_transforms(chain, angles)
                    eff = T_list[-1][:3, 3]
                    log_joint_angles(chain, angles, step)
                    log_status(step, g, eff, 0.0, eff)
                    step += 1
                    time.sleep(1.0 / 30.0)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        # Fixed goals mode
        goals = np.array(args.goals, dtype=float).reshape(-1, 3)
        for g in goals:
            angles, step = move_to_goal(g, step)

    if args.sleep > 0:
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()

