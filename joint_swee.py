import placo
import rerun as rr
from rerun.archetypes import Transform3D
import numpy as np
import placo
from matplotlib import pyplot as plt
def generate_n_dim_coords_linspace(bounds, num_points):
    """
    Generates all coordinates in an n-dimensional space given bounds and number of points.

    Args:
        bounds (list of tuples): A list of (min, max) for each dimension.
        num_points (int or list of int): The number of points along each dimension.

    Returns:
        numpy.ndarray: An array where each row is a coordinate point.
    """
    # Create 1D coordinate vectors for each dimension
    if isinstance(num_points, int):
        num_points = [num_points] * len(bounds)
    
    ranges = [np.linspace(b[0], b[1], n) for b, n in zip(bounds, num_points)]
    
    # Use meshgrid to create N-D coordinate arrays
    # indexing='ij' ensures the order matches the input bounds
    coords_nd = np.meshgrid(*ranges, indexing='ij')
    
    # Stack the N-D arrays along a new axis and reshape into a list of points
    coords_points = np.stack(coords_nd, axis=-1).reshape(-1, len(bounds))
    
    return coords_points

fig = plt.figure(figsize=(14, 4))

coordaxes = fig.add_subplot(131, projection="3d")
xy_axes    = fig.add_subplot(132)
xz_axes    = fig.add_subplot(133)

fig.tight_layout()

# Loading a robot
print("Loading Robot URDF")
robot = placo.RobotWrapper("../../SO101", placo.Flags.ignore_collisions )

# Initializing the kinematics solver
print("Initializing Solver and Palco")
solver = placo.KinematicsSolver(robot)
# The floating base can't move
solver.mask_fbase(True)
# j = robot.get_joint_limits("Proximal")

# Adding a frame task
effector_task = solver.add_position_task("gripper_frame_link", np.zeros(3))
# effector_task.mask_rotation(True)
J = robot.frame_jacobian("gripper_frame_link", "world")[:3, :]  # position part


total_rots = generate_n_dim_coords_linspace([(-0.0001, 0.0001 ), (-3.14 / 2, 3.14 /2 ), (-3.14 / 2, 3.14 /2 ), (-3.14 / 2, 3.14 /2 ), (-3.14 / 2, 3.14 /2 )], num_points= [1, 10, 10, 10, 1])

run_count = len(total_rots)
good_coords = []
bad_coords = []
for xx in range(run_count):
    if xx % 10000 == 0:
        print(f"Processed {100 * xx / run_count:.2f} %")
    robot.reset()
    joint_rots = total_rots[xx]
    joint_rots = np.append(joint_rots, 0) # account for gripper DOF
    joint_names = robot.joint_names()
    
    for x in range(len(robot.joint_names())):
        robot.set_joint(joint_names[x], 
                        joint_rots[x])
    robot.update_kinematics()
    pos = robot.get_T_world_frame("gripper_frame_link")[:3, 3]

    good_coords.append(pos)
print(f"Finished random {run_count} points tested")
print("Rendering points")
good_coords = np.array(good_coords)
bad_coords = np.array(bad_coords)

print(good_coords.shape)
alph = 0.1
coordaxes.scatter(good_coords[:, 0], good_coords[:, 1], good_coords[:, 2], marker="o", c="green", alpha=alph)

xy_axes.scatter(good_coords[:, 0], good_coords[:, 1], marker="o", c="green", alpha=alph)
xy_axes.spines['left'].set_position("center")
xy_axes.spines['bottom'].set_position("center")
xz_axes.scatter(good_coords[:, 0], good_coords[:, 2], marker="o", c="green", alpha=alph)
xz_axes.spines['left'].set_position("center")
xz_axes.spines['bottom'].set_position("center")
print("Points rendered")

# 3D
coordaxes.set_xlabel("X")
coordaxes.set_ylabel("Y")
coordaxes.set_zlabel("Z")
coordaxes.set_title("XYZ")

# XY
xy_axes.set_xlabel("X")
xy_axes.set_ylabel("Y")
xy_axes.set_title("XY Projection")
xy_axes.axis("equal")

# XZ
xz_axes.set_xlabel("X")
xz_axes.set_ylabel("Z")
xz_axes.set_title("XZ Projection")
xz_axes.axis("equal")

plt.show()