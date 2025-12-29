import rerun as rr
from kinematics import RobotKinematics
from rendering.rerun_render import RerunViewer
from planner.prm_planner import PRMPlanner
import numpy as np
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--connect", action="store_true")
args = parser.parse_args()
connect = args.connect

def generate_random_goal(arm_reach: float = 0.25):
    r_min, r_max = 0.2, 0.3

    theta = np.random.uniform(-np.pi/2, np.pi/2)

    phi_min = 0.2
    phi_max = np.pi * 0.6
    u = np.random.uniform(np.cos(phi_max), np.cos(phi_min))
    phi = np.arccos(u)

    r = ((r_max**3 - r_min**3) * np.random.random() + r_min**3) ** (1/3)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi) + 0.05

    return phi, np.array([x, y, z])


rr.init("total_view", spawn= not connect)
if connect :
    rr.connect_grpc("rerun+http://172.18.128.1:9876/proxy")
    
robot_renderer = RerunViewer("rerun_arm/robot.urdf")
kinematics = RobotKinematics("rerun_arm", "gripper_frame_link")

joints_task = kinematics.solver.add_joints_task()
joints_task.set_joints({
    "Pan": 0,
    "Proximal": 0.0,
    "Distal": 3.14/2,
    "Wrist": 0,
    "Roll": 0,
    "Gripper": 0,
})
joints_task.configure("joints_regularization", "soft", 1e-2)

# Create PRM planner with smoother interpolation
planner = PRMPlanner(num_samples=2000, k_neighbors=15, edge_resolution=5, interpolation_steps=30)

# Set workspace to match goal generation (slightly larger for buffer)
planner.set_workspace(
    r_min=0.15, r_max=0.35,      # Goal uses 0.2-0.3, we go wider
    theta_min=-np.pi/2, theta_max=np.pi/2,
    phi_min=0.1, phi_max=np.pi * 0.7,  # Goal uses 0.2 to pi*0.6
    z_offset=0.05
)

# Try to load existing roadmap, otherwise build new one
roadmap_path = Path("../../rerun_arm/roadmap.json")
if not planner.load_roadmap(str(roadmap_path)):
    planner.build_roadmap(kinematics, verbose=True)
    planner.save_roadmap(str(roadmap_path))

frame = 0
for x in range(50):
    frame += 1
    rr.set_time("frame", sequence=frame)
    current_pos = kinematics.get_ee_pos()
    (_, goal) = generate_random_goal()
    rr.log("points", rr.Points3D(goal, colors=[255, 255, 255], radii=0.01), rr.CoordinateFrame("base"))

    # Plan path using joint-space interpolation
    path = planner.generate_path(kinematics, current_pos, goal)

    if path is None:
        # Planning failed - mark goal as red and show intended straight line
        vector = goal - current_pos
        rr.log("movement_path", rr.Arrows3D(origins=current_pos, vectors=vector, colors=[255, 0, 0]), rr.CoordinateFrame("base"))
        rr.log("points", rr.Points3D(goal, colors=[255, 0, 0], radii=0.01), rr.CoordinateFrame("base"))
        print(f"Planning failed for goal {goal}")
        continue

    # Convert joint path to Cartesian path for visualization
    cartesian_path = planner.get_cartesian_path(kinematics, path)
    rr.log("movement_path", rr.LineStrips3D([cartesian_path], colors=[0, 255, 0]), rr.CoordinateFrame("base"))

    # Execute path
    for joints in path:
        frame += 1
        rr.set_time("frame", sequence=frame)
        robot_renderer.write_joint_positions(list(joints))

    # Update robot state to final position so next iteration starts from here
    final_joints = path[-1]
    for idx, name in enumerate(kinematics.robot.joint_names()):
        kinematics.robot.set_joint(name, final_joints[idx])
    kinematics.robot.update_kinematics()

    # Mark goal as green (reached)
    rr.log("points", rr.Points3D(goal, colors=[0, 255, 0], radii=0.01), rr.CoordinateFrame("base"))