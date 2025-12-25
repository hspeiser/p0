#!/usr/bin/env python3
"""
Load an external URDF (with package://assets) and display it in Rerun.

Usage:
  python load_external_urdf.py --urdf /Users/leonardspeiser/Desktop/Archive/robot.urdf --sleep 8

Notes:
  - Sets RERUN_URDF_PACKAGE_PATH to the URDF's parent dir so package://assets/... meshes resolve.
  - Spawns a viewer by default; keep the script alive briefly (--sleep) so you see the model.
  - Run inside your venv:
      cd /Users/leonardspeiser/Projects/rerun
      source .venv/bin/activate
      python examples/python/robot_arm/load_external_urdf.py --urdf /path/to/robot.urdf --sleep 8

"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import rerun as rr

if TYPE_CHECKING:
    from collections.abc import Callable


def find_asset_refs(urdf_path: Path) -> list[str]:
    """Extract package://assets/... references from the URDF text."""
    refs: list[str] = []
    text = urdf_path.read_text(errors="ignore", encoding="utf-8")
    needle = "package://assets/"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        end = idx + len(needle)
        # read until quote or whitespace
        while end < len(text) and text[end] not in ['"', "'", " ", "\n", "\r", "\t", "<"]:
            end += 1
        refs.append(text[idx + len("package://") : end])
        start = end
    return refs


@dataclass
class JointSpec:
    name: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


def parse_joints(urdf_path: Path) -> list[JointSpec]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints: list[JointSpec] = []
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link", "")
        child = child_el.get("link", "")
        origin_el = joint.find("origin")
        origin_xyz: tuple[float, float, float]
        origin_rpy: tuple[float, float, float]
        axis: tuple[float, float, float]
        if origin_el is not None:
            xyz_list = [float(x) for x in origin_el.get("xyz", "0 0 0").split()][:3]
            while len(xyz_list) < 3:
                xyz_list.append(0.0)
            rpy_list = [float(r) for r in origin_el.get("rpy", "0 0 0").split()][:3]
            while len(rpy_list) < 3:
                rpy_list.append(0.0)
            origin_xyz = (xyz_list[0], xyz_list[1], xyz_list[2])
            origin_rpy = (rpy_list[0], rpy_list[1], rpy_list[2])
        else:
            origin_xyz = (0.0, 0.0, 0.0)
            origin_rpy = (0.0, 0.0, 0.0)
        axis_el = joint.find("axis")
        if axis_el is not None:
            axis_list = [float(a) for a in axis_el.get("xyz", "0 0 1").split()][:3]
            while len(axis_list) < 3:
                axis_list.append(0.0)
            axis = (axis_list[0], axis_list[1], axis_list[2])
        else:
            axis = (0.0, 0.0, 1.0)
        joint_type = joint.get("type", "fixed")
        # Only animate revolute/continuous
        if joint_type not in ("revolute", "continuous"):
            continue
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


def log_urdf_static(urdf_path: Path, package_path: Path) -> None:
    """Log the URDF once, resolving package://assets via package_path."""
    os.environ["RERUN_URDF_PACKAGE_PATH"] = str(package_path)
    rr.log_file_from_path(urdf_path, static=True)


def get_joint_logger(
    urdf_path: Path, package_path: Path | None = None
) -> tuple[list[JointSpec], Callable[[int, dict[str, float] | list[float]], None]]:
    """
    Load URDF, parse joints, and return (joints, log_fn).

    log_fn(step: int, angles: dict[str, float] | list[float]) will log transforms
    for each joint using the URDF origins/axes. If a list is passed, it uses the
    joint order from the URDF.
    """
    package_path = package_path or urdf_path.parent
    os.environ["RERUN_URDF_PACKAGE_PATH"] = str(package_path)

    # Rewrite URDF package://assets/... to file://...
    text = urdf_path.read_text(encoding="utf-8")
    assets_abs = (package_path / "assets").resolve()
    prefix = "package://assets/"
    replacement = f"file://{assets_abs}/"
    if prefix in text:
        rewritten = text.replace(prefix, replacement)
        tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
        tmp.write(rewritten.encode("utf-8"))
        tmp.flush()
        tmp.close()
        urdf_to_load = Path(tmp.name)
    else:
        urdf_to_load = urdf_path

    joints = parse_joints(urdf_to_load)

    def log_fn(step: int, angles: dict[str, float] | list[float]) -> None:
        rr.set_time("step", sequence=step)
        for idx, j in enumerate(joints):
            if isinstance(angles, dict):
                if j.name not in angles:
                    continue
                angle = float(angles[j.name])
            else:
                if idx >= len(angles):
                    continue
                angle = float(angles[idx])
            rot_origin = euler_to_quat(j.origin_rpy)
            rot_dyn = axis_angle_to_quat(j.axis, angle)
            rot = quat_mul(rot_origin, rot_dyn)
            rr.log(
                f"world/urdf/joints/{j.name}",
                rr.Transform3D(
                    translation=list(j.origin_xyz),
                    rotation=rot,
                    parent_frame=j.parent,
                    child_frame=j.child,
                ),
            )

    return joints, log_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an external URDF and view it in Rerun.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/Users/leonardspeiser/Desktop/Archive/robot.urdf"),
        help="Path to the URDF file",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=8.0,
        help="Seconds to keep the script alive after loading (so the viewer stays visible)",
    )
    parser.add_argument(
        "--package-path",
        type=Path,
        default=None,
        help="Override RERUN_URDF_PACKAGE_PATH (defaults to URDF parent)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate joints with simple sine waves to verify motion",
    )
    parser.add_argument(
        "--keep-static-first",
        action="store_true",
        help="Log the static URDF once before animating (keeps initial pose visible in tree).",
    )
    args = parser.parse_args()

    urdf_path: Path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Ensure package://assets resolves.
    package_path = args.package_path.expanduser().resolve() if args.package_path else urdf_path.parent
    os.environ["RERUN_URDF_PACKAGE_PATH"] = str(package_path)
    assets_dir = package_path / "assets"

    # Spawn viewer explicitly so it opens without CLI flags.
    rr.init("rerun_external_urdf", spawn=True)

    # Align camera & coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Log a visible origin marker and axes under world/
    rr.log(
        "world/debug/origin",
        rr.Boxes3D(
            half_sizes=[[0.2, 0.2, 0.2]],
            centers=[[0.0, 0.0, 0.0]],
            colors=[[255, 0, 0, 200]],
        ),
        static=True,
    )
    rr.log(
        "world/debug/axes",
        rr.Arrows3D(
            origins=[[0.0, 0.0, 0.0]] * 3,
            vectors=[
                [0.3, 0.0, 0.0],  # X red
                [0.0, 0.3, 0.0],  # Y green
                [0.0, 0.0, 0.3],  # Z blue
            ],
            colors=[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            radii=[0.01],
        ),
        static=True,
    )

    print(f"Loading URDF: {urdf_path}")
    print(f"RERUN_URDF_PACKAGE_PATH={os.environ['RERUN_URDF_PACKAGE_PATH']}")
    if assets_dir.is_dir():
        count = sum(1 for _ in assets_dir.rglob("*.stl"))
        print(f"Found assets dir: {assets_dir} with {count} STL files")
    else:
        print(f"Warning: assets dir not found at {assets_dir}")

    refs = find_asset_refs(urdf_path)
    missing = []
    for ref in refs:
        # ref like assets/foo.stl
        candidate = Path(os.environ["RERUN_URDF_PACKAGE_PATH"]) / ref.replace("assets/", "assets/", 1)
        if not candidate.is_file():
            missing.append(candidate)
    if missing:
        print("Missing referenced asset files:")
        for m in missing[:20]:
            print(f"  - {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("All referenced assets found (based on text scan).")

    # Rewrite URDF package://assets/... to absolute file://... to avoid ROS package lookup failures.
    rewritten_path = None
    try:
        text = urdf_path.read_text(encoding="utf-8")
        assets_abs = assets_dir.resolve()
        # Ensure trailing slash
        prefix = "package://assets/"
        replacement = f"file://{assets_abs}/"
        if prefix in text:
            rewritten = text.replace(prefix, replacement)
            tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
            tmp.write(rewritten.encode("utf-8"))
            tmp.flush()
            tmp.close()
            rewritten_path = Path(tmp.name)
            print(f"Rewrote URDF to {rewritten_path} with assets -> {replacement}")
        else:
            print("No package://assets/ references found to rewrite.")
    except Exception as e:
        print(f"Warning: failed to rewrite URDF: {e}")
        rewritten_path = None

    urdf_to_load = rewritten_path or urdf_path

    # Log the URDF; static=True so geometry is loaded once.
    rr.log_file_from_path(urdf_to_load, static=True)

    # If requested, keep the initial pose visible (helpful for comparing rest pose)
    if args.keep_static_first and args.animate:
        rr.log("world/urdf/static_rest_pose", rr.Clear(recursive=True), static=True)
        rr.log_file_from_path(urdf_to_load, static=True)

    # Optional simple animation using joint origins/axes from URDF
    if args.animate:
        joints = parse_joints(urdf_to_load)
        if not joints:
            print("No revolute joints found to animate.")
        else:
            steps = 400
            for step in range(steps):
                t = step / 40.0  # adjust speed
                rr.set_time("step", sequence=step)
                for j_idx, j in enumerate(joints):
                    # Small angles to avoid breaking the arm apart
                    angle = 0.25 * math.sin(t * (0.6 + 0.2 * j_idx))
                    rot_origin = euler_to_quat(j.origin_rpy)
                    rot_dyn = axis_angle_to_quat(j.axis, angle)
                    rot = quat_mul(rot_origin, rot_dyn)
                    rr.log(
                        f"world/urdf/joints/{j.name}",
                        rr.Transform3D(
                            translation=list(j.origin_xyz),
                            rotation=rot,
                            parent_frame=j.parent,
                            child_frame=j.child,
                        ),
                    )
            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        # Keep the script alive briefly so the viewer shows the model.
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
