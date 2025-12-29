import math
import numpy as np
import rerun as rr
import xml.etree.ElementTree as ET
from pathlib import Path


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    axis = axis / norm
    s = math.sin(angle / 2)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)],
        dtype=np.float32,
    )


def quat_from_euler_xyz(rpy):
    roll, pitch, yaw = rpy
    qx = quat_from_axis_angle([1, 0, 0], roll)
    qy = quat_from_axis_angle([0, 1, 0], pitch)
    qz = quat_from_axis_angle([0, 0, 1], yaw)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


class JointInfo:
    def __init__(self, name, parent, child, origin_xyz, origin_rpy, axis, joint_type):
        self.name = name
        self.parent = parent
        self.child = child
        self.origin_xyz = origin_xyz
        self.origin_rpy = origin_rpy
        self.axis = axis
        self.joint_type = joint_type


def parse_urdf_joints(urdf_path):
    """Parse URDF and extract joint information."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []

    for joint in root.findall("joint"):
        joint_type = joint.get("type", "fixed")
        name = joint.get("name", "")

        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue

        parent = parent_el.get("link", "")
        child = child_el.get("link", "")

        # Parse origin
        origin_el = joint.find("origin")
        if origin_el is not None:
            xyz_str = origin_el.get("xyz", "0 0 0")
            rpy_str = origin_el.get("rpy", "0 0 0")
            origin_xyz = [float(x) for x in xyz_str.split()][:3]
            origin_rpy = [float(x) for x in rpy_str.split()][:3]
            while len(origin_xyz) < 3:
                origin_xyz.append(0.0)
            while len(origin_rpy) < 3:
                origin_rpy.append(0.0)
        else:
            origin_xyz = [0.0, 0.0, 0.0]
            origin_rpy = [0.0, 0.0, 0.0]

        # Parse axis
        axis_el = joint.find("axis")
        if axis_el is not None:
            axis_str = axis_el.get("xyz", "0 0 1")
            axis = [float(a) for a in axis_str.split()][:3]
            while len(axis) < 3:
                axis.append(0.0)
        else:
            axis = [0.0, 0.0, 1.0]

        joints.append(JointInfo(
            name=name,
            parent=parent,
            child=child,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis,
            joint_type=joint_type
        ))

    return joints


class RerunViewer:
    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path)
        rr.log_file_from_path(str(self.urdf_path), static=True)
        self.joints = parse_urdf_joints(self.urdf_path)
        # Filter to only revolute/continuous joints
        self.revolute_joints = [j for j in self.joints if j.joint_type in ("revolute", "continuous")]

    def write_joint_positions(self, joint_positions):
        """Update joint transforms in rerun.

        joint_positions: list of angles in radians, one per revolute joint
        """
        for joint_index, joint in enumerate(self.revolute_joints):
            if joint_index >= len(joint_positions):
                continue

            # Map from the end (gripper) backwards
            angle = joint_positions[len(joint_positions) - 1 - joint_index]

            # Compute rotation: origin rotation * axis rotation
            q_origin = quat_from_euler_xyz(joint.origin_rpy)
            q_axis = quat_from_axis_angle(joint.axis, angle)
            rotation = quat_mul(q_origin, q_axis)

            rr.log(
                "transforms",
                rr.Transform3D(
                    translation=joint.origin_xyz,
                    quaternion=rr.Quaternion(xyzw=rotation.tolist()),
                    parent_frame=joint.parent,
                    child_frame=joint.child,
                ),
            )


if __name__ == "__main__":
    rr.init("rerun_example_animated_urdf", spawn=True)

    urdf_path = "../../rerun_arm/robot.urdf"
    viewer = RerunViewer(urdf_path)

    # Test animation
    for step in range(1000):
        rr.set_time("frame", sequence=step)
        # Sine wave for each joint
        angles = [0.5 * math.sin(step * 0.05 + i) for i in range(6)]
        viewer.write_joint_positions(angles)
