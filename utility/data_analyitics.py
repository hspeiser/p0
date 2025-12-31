import placo
import rerun as rr
from rerun.archetypes import Transform3D
import numpy as np
import placo
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(14, 4))

coordaxes = fig.add_subplot(131, projection="3d")
xy_axes    = fig.add_subplot(132)
xz_axes    = fig.add_subplot(133)

fig.tight_layout()
def generate_random_goal(arm_reach: float = 0.25):
    r_min, r_max = 0.1, 0.3

    theta = np.random.uniform(0, 2 * np.pi)

    phi_min = 0.2
    phi_max = np.pi * 0.6
    u = np.random.uniform(np.cos(phi_max), np.cos(phi_min))
    phi = np.arccos(u)

    r = ((r_max**3 - r_min**3) * np.random.random() + r_min**3) ** (1/3)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi) + 0.05

    return phi, np.array([x, y, z])

# Loading a robot
print("Loading Robot URDF")
robot = placo.RobotWrapper("../../rerun_arm/", placo.Flags.ignore_collisions )

# Initializing the kinematics solver
print("Initializing Solver and Palco")
solver = placo.KinematicsSolver(robot)
# The floating base can't move
solver.mask_fbase(True)
j = robot.get_joint_limits("Proximal")

# Adding a frame task
effector_task = solver.add_position_task("gripper_frame_link", np.zeros(3))
# effector_task.mask_rotation(True)
J = robot.frame_jacobian("gripper_frame_link", "world")[:3, :]  # position part

random = True

coords = []
epsilon = .001
run_count =1000000
if not random:
    
    min = -.5
    max = 0.5
    step = 0.05
    print(f"Going through grid from {min} to {max} in steps of {step}")
    X, Y, Z = np.mgrid[min:max:step, min:max:step, 0:max * 1.5 :step]
    coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    run_count = len(coords)
else:
    print(f"Doing {run_count} random points")
good_coords = []
bad_coords = []

resting_position = [0.0, 0.0, 3.14/2, 0, 0, 0]

for xx in range(run_count):
    if xx % 1000 == 0:
        print(f"Processed {100 * xx / run_count:.2f} %")
    # robot.reset()
    for (idx, name) in enumerate(robot.joint_names()):
        robot.set_joint(name, resting_position[idx])
    goal = None
    if random:
        (phi , goal) = generate_random_goal(arm_reach=.3)
    else:
        goal = coords[xx]
    effector_task.target_world = goal
    # Updating kinematics computations (frames, jacobians, etc.)
    robot.update_kinematics()
    # Solving the IK
    for x in range(25):
        solver.solve(True)
        robot.update_kinematics()
    pos = robot.get_T_world_frame("gripper_frame_link")[:3, 3]

    residual = np.linalg.norm(goal - pos)
    if residual < epsilon:
        good_coords.append(goal)
    else:
        bad_coords.append(goal)
for xx in range(run_count):
    if xx % 1000 == 0:
        print(f"Processed {100 * xx / run_count:.2f} %")
    robot.reset()
    goal = None
    if random:
        (phi , goal) = generate_random_goal(arm_reach=.3)
    else:
        goal = coords[xx]
    effector_task.target_world = goal
    # Updating kinematics computations (frames, jacobians, etc.)
    robot.update_kinematics()
    # Solving the IK
    for x in range(25):
        solver.solve(True)
        robot.update_kinematics()
    pos = robot.get_T_world_frame("gripper_frame_link")[:3, 3]

    residual = np.linalg.norm(goal - pos)
    if residual < epsilon:
        good_coords.append(goal)
    else:
        bad_coords.append(goal)
print(f"Finished random {run_count} points tested")
print("Rendering points")
good_coords = np.unique(np.round(good_coords, 4), axis=0)
bad_coords = np.unique(np.round(bad_coords, 4), axis=0)

print(good_coords.shape)

coordaxes.scatter(good_coords[:, 0], good_coords[:, 1], good_coords[:, 2], marker="o", c="green", alpha=0.1)

xy_axes.scatter(good_coords[:, 0], good_coords[:, 1], marker="o", c="green", alpha=0.1)

xz_axes.scatter(good_coords[:, 0], good_coords[:, 2], marker="o", c="green", alpha=0.1)
# for coord in bad_coords:
coordaxes.scatter(bad_coords[:, 0], bad_coords[:, 1], bad_coords[:, 2], marker="x", c="red", alpha=epsilon)

xy_axes.scatter(bad_coords[:, 0], bad_coords[:, 1], marker="x", c="red", alpha=epsilon)

xz_axes.scatter(bad_coords[:, 0], bad_coords[:, 2], marker="x", c="red", alpha=epsilon)
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