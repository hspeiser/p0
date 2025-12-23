import rerun as rr
import numpy as np
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R

def remap(value, from_min, from_max, to_min, to_max):
    """Remap a value from one range to another"""
    return to_min + (value - from_min) * (to_max - to_min) / (from_max - from_min)

def main():
    urdf_path = r"C:\Users\satas\p0\SO101\so101_new_calib.urdf"
    
    # Initialize Rerun
    rec = rr.RecordingStream("rerun_example_animated_urdf")
    rec.spawn()
    
    # Log the URDF file once, as a static resource
    rec.log_file_from_path(urdf_path, static=True)
    
    # Load the URDF tree structure into memory
    robot = URDF.from_xml_file(urdf_path)
    
    # Animate
    for step in range(10000):
        rec.set_time_sequence("step", step)
        
        for joint_index, joint in enumerate(robot.joints):
            if joint.joint_type == 'revolute':
                fixed_axis = np.array(joint.axis)
                
                # Usually this angle would come from a measurement - here we just fake something
                sin_value = np.sin(step * (0.02 + joint_index / 100.0))
                dynamic_angle = remap(
                    sin_value,
                    -1.0, 1.0,
                    joint.limit.lower, joint.limit.upper
                )
                
                # Compute the full rotation for this joint
                # First apply origin RPY rotation
                origin_rotation = R.from_euler('xyz', joint.origin.rpy)
                
                # Then apply the dynamic joint rotation around the axis
                axis_rotation = R.from_rotvec(dynamic_angle * fixed_axis)
                
                # Combine rotations
                combined_rotation = origin_rotation * axis_rotation
                quat = combined_rotation.as_quat()  # [x, y, z, w]
                
                # Rerun loads the URDF transforms with child/parent frame relations.
                # In order to move a joint, we just need to log a new transform between two of those frames.
                rec.log(
                    "/transforms",
                    rr.Transform3D(
                        translation=joint.origin.xyz,
                        rotation=rr.Quaternion(xyzw=[quat[0], quat[1], quat[2], quat[3]]),
                        relation=rr.TransformRelation.ChildFromParent,
                    ).with_parent(joint.parent).with_child(joint.child)
                )

if __name__ == "__main__":
    main()