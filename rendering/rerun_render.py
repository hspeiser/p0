import math
import numpy as np
import rerun as rr
import yourdfpy
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


class RerunViewer:
    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path)
        rr.log_file_from_path(str(self.urdf_path), static=True)
        
        # Load URDF using yourdfpy
        self.robot = yourdfpy.URDF.load(str(self.urdf_path))
        
        # Filter to only revolute/continuous joints
        self.revolute_joints = [
            joint for joint in self.robot.joint_map.values()
            if joint.type in ("revolute", "continuous")
        ]

    def write_joint_positions(self, joint_positions):
        """Update joint transforms in rerun.

        joint_positions: list of angles in radians, one per revolute joint
        """
        for joint_index, joint in enumerate(self.revolute_joints):
            if joint_index >= len(joint_positions):
                continue

            # Map from the end (gripper) backwards
            angle = joint_positions[len(joint_positions) - 1 - joint_index]

            # Extract origin translation and rotation
            origin_xyz = joint.origin[:3, 3].tolist() if joint.origin is not None else [0.0, 0.0, 0.0]
            
            # Extract RPY from origin rotation matrix
            if joint.origin is not None:
                rotation_matrix = joint.origin[:3, :3]
                # Convert rotation matrix to RPY (roll, pitch, yaw)
                sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
                singular = sy < 1e-6
                if not singular:
                    roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    pitch = math.atan2(-rotation_matrix[2, 0], sy)
                    yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                    pitch = math.atan2(-rotation_matrix[2, 0], sy)
                    yaw = 0
                origin_rpy = [roll, pitch, yaw]
            else:
                origin_rpy = [0.0, 0.0, 0.0]

            # Get joint axis
            axis = joint.axis.tolist() if joint.axis is not None else [0.0, 0.0, 1.0]

            # Compute rotation: origin rotation * axis rotation
            q_origin = quat_from_euler_xyz(origin_rpy)
            q_axis = quat_from_axis_angle(axis, angle)
            rotation = quat_mul(q_origin, q_axis)

            rr.log(
                "transforms",
                rr.Transform3D(
                    translation=origin_xyz,
                    quaternion=rr.Quaternion(xyzw=rotation.tolist()),
                    parent_frame=joint.parent,
                    child_frame=joint.child,
                ),
            )


if __name__ == "__main__":
    rr.init("robot_workspace_viewer", spawn=False)
    rr.connect_grpc("rerun+http://172.18.128.1:9876/proxy")

    urdf_path = "rerun_arm/robot.urdf"
    viewer = RerunViewer(urdf_path)

    # Test animation
    for step in range(1000):
        rr.set_time("frame", sequence=step)
        # Sine wave for each joint
        angles = [0.5 * math.sin(step * 0.05 + i) for i in range(6)]
        viewer.write_joint_positions(angles)