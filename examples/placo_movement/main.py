import placo
import rerun as rr
from rerun.archetypes import Transform3D
import numpy as np
import placo
from matplotlib import pyplot as plt

def generate_random_goal(arm_reach: float = 0.25):
    """Generate a random reachable goal within the arm's workspace."""
    # Random point in a sphere, biased toward reachable positions
    r = (0.1 + (.3 - 0.1) * np.random.random())  # 30%-100% of reach
    theta = np.random.uniform(0, 2 * np.pi)  # Azimuth
    phi = np.random.uniform(0.2, np.pi * 0.6)  # Elevation (avoid straight up/down)

    x = max(r * np.sin(phi) * np.cos(theta), 0)

    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi) + 0.05  # Offset up from base

    return (phi,  np.array([x, y, z]))
# Loading a robot
robot = placo.RobotWrapper("../../rerun_arm/", placo.Flags.ignore_collisions )
# Initializing the kinematics solver
solver = placo.KinematicsSolver(robot)
# The floating base can't move
solver.mask_fbase(True)


# Adding a frame task
effector_task = solver.add_position_task("gripper_frame_link", np.zeros(3))
# effector_task.mask_rotation(True)
J = robot.frame_jacobian("gripper_frame_link", "world")[:3, :]  # position part
print("Jacobian rank:", np.linalg.matrix_rank(J))
print("DOF:", J.shape[1])
# regularization_task = solver.add_regularization_task(1e-4)
# regularization_task.set_joint_weight("Pan", 1e2)
from placo_utils.tf import tf
# Updating the target pose of the effector task


# print(robot.get_T_world_frame("base")[:3,3])

for zz in range(1, 2):
    z_scale = 25
    zz = zz * z_scale
    task_weights = []
    residuals = []  
    for yy in range(0,100, 1):
        print(yy)
        y_scale = 0.01
        yy = yy  * y_scale
        # effector_task.configure("effector", "soft", 100000000)

        total_residual = 0
        run_count = 300
        for xx in range(run_count):
            
            robot.reset()
            (phi , goal) = generate_random_goal(arm_reach=.3)

            effector_task.target_world = goal
            # Updating kinematics computations (frames, jacobians, etc.)
            robot.update_kinematics()
            # Solving the IK
            for x in range(25):
                solver.solve(True)
                robot.update_kinematics()
            pos = robot.get_T_world_frame("gripper_frame_link")[:3, 3]

            residual = np.linalg.norm(goal - pos)
            if residual < 0.01:
                prox_joint = robot.get_joint("Wrist")
                dist_joint = robot.get_joint("Distal")
                residuals.append(residual)
                task_weights.append(prox_joint)
    plt.scatter(task_weights, residuals, label = f"{zz}", alpha = 0.01)
    print(zz)
# plt.ylim(bottom=0)
# plt.semilogy()
plt.legend()
plt.savefig("thingle.png")
# print(total_residual / 100)
# solver.dump_status()
# print(robot.state.q)
# names = ["Pan", "Proximal", "Distal", "Wrist", "Roll", "Gripper"]
# for name in names:
#     print(name, ": ", robot.get_joint(name))
# T_world_jaw = robot.get_T_world_frame("jaw")
# position = T_world_jaw[:3, 3]

# print("End-effector position (world):", position)
# print("End effector goal            :", goal)