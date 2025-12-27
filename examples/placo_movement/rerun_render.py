import math
import numpy as np
import rerun as rr

from rerun.archetypes import Transform3D
from rerun.components import Translation3D
from urdfpy import URDF, matrix_to_xyz_rpy
# -------------------------------
# Helpers
# -------------------------------

def remap(x, in_min, in_max, out_min, out_max):
    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)


def quat_from_euler_xyz(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    return np.array([x, y, z, w], dtype=np.float32)


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)],
        dtype=np.float32,
    )


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)
def quat_from_euler_xyz_extrinsic(rpy):
    roll, pitch, yaw = rpy

    qx = quat_from_axis_angle([1, 0, 0], roll)
    qy = quat_from_axis_angle([0, 1, 0], pitch)
    qz = quat_from_axis_angle([0, 0, 1], yaw)

    # NOTE: reversed order
    return quat_mul(qz, quat_mul(qy, qx))
# -------------------------------
# Main
# -------------------------------

def main():
    rr.init("rerun_example_animated_urdf")
    
    # Remove or comment out the grpc connection if not available
    rr.connect_grpc("rerun+http://172.18.128.1:9876/proxy")
    
    urdf_path = "../../rerun_arm/robot.urdf"  # Update path to your URDF file
    try:
        rr.log_file_from_path(urdf_path, static=True)
    except Exception as e:
        print(f"Warning: Could not log URDF file: {e}")
    print("loading urdf")
    try:
        robot = URDF.load(urdf_path)
    except FileNotFoundError:
        print(f"Error: URDF file not found at {urdf_path}")
        return
    print("loaded urdf")
    for step in range(10_000):
        rr.set_time("frame", sequence=step)

        for joint_index, joint in enumerate(robot.joints):
            if joint.joint_type != "revolute":
                continue

            # Handle cases where axis might be None


            axis = joint.axis
            phase = step * (0.02 + joint_index / 100.0)

            # Handle cases where limits might not be set
            if joint.limit is None:
                lower, upper = -math.pi, math.pi
            else:
                lower, upper = joint.limit.lower, joint.limit.upper

            angle = remap(
                math.sin(phase),
                -1.0, 1.0,
                lower,
                upper,
            )
            joint_rpy = matrix_to_xyz_rpy(joint.origin)[3:6]
            joint_xyz = matrix_to_xyz_rpy(joint.origin)[0:3]
            q_origin = quat_from_euler_xyz_extrinsic(joint_rpy)
            q_axis = quat_from_axis_angle(axis, angle)
            rotation = quat_mul(q_origin, q_axis)

            rr.log(
                f"transforms",
                Transform3D(
                    rotation=rr.Quaternion(xyzw=rotation),
                    translation=Translation3D(joint_xyz),
                    parent_frame=joint.parent,
                    child_frame=joint.child,
                ),
            )
if __name__ == "__main__":
    main()

class RerunViewer:
    def __init__(self, urdf_path):
        rr.log_file_from_path(urdf_path, static=True)
        self.robot = URDF.load(urdf_path)
    def write_joint_positions(self, joint_positions):
        for joint_index, joint in enumerate(self.robot.joints):
            if joint.joint_type != "revolute":
                continue
            
            axis = joint.axis
            
            angle = joint_positions[len(joint_positions) - 1 -joint_index]
            joint_rpy = matrix_to_xyz_rpy(joint.origin)[3:6]
            joint_xyz = matrix_to_xyz_rpy(joint.origin)[0:3]
            q_origin = quat_from_euler_xyz_extrinsic(joint_rpy)
            q_axis = quat_from_axis_angle(axis, angle)
            rotation = quat_mul(q_origin, q_axis)
            rr.log(
                f"transforms",
                Transform3D(
                    rotation=rr.Quaternion(xyzw=rotation),
                    translation=Translation3D(joint_xyz),
                    parent_frame=joint.parent,
                    child_frame=joint.child,
                ),
            )