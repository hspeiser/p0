import rerun as rr
from kinematics import RobotKinematics
from rerun_render import RerunViewer
import numpy as np
import time
def generate_random_goal(arm_reach: float = 0.25):
    r_min, r_max = 0.2, 0.3

    theta = np.random.uniform(-np.pi/4, np.pi/4)

    phi_min = 0.2
    phi_max = np.pi * 0.6
    u = np.random.uniform(np.cos(phi_max), np.cos(phi_min))
    phi = np.arccos(u)

    r = ((r_max**3 - r_min**3) * np.random.random() + r_min**3) ** (1/3)

    x = max(r * np.sin(phi) * np.cos(theta), 0)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi) + 0.05

    return phi, np.array([x, y, z])
rr.init("total_view")
rr.connect_grpc("rerun+http://172.18.128.1:9876/proxy")
robot_renderer = RerunViewer("../../rerun_arm/robot.urdf")
kinematics = RobotKinematics("../../rerun_arm", "gripper_frame_link")
total_points = []
frame = 0
for x in range(100):
    frame += 1
    rr.set_time("frame", sequence=frame)
    current_pos = kinematics.get_ee_pos()
    (_, goal) = generate_random_goal()
    rr.log(f"points", rr.Points3D(goal, colors=[255,255,255], radii=.01), rr.CoordinateFrame("base"))

    vector = goal - current_pos
    rr.log(f"movement_path", rr.Arrows3D(origins=current_pos, vectors=vector), rr.CoordinateFrame("base"))
    steps = 100
    for step in range(steps):
        rr.set_time("frame", sequence=frame)
        frame += 1
        new_goal = vector * (1/steps) *  (step + 1) + current_pos
        (converged, joint_pos) = kinematics.inverse_kinematics(new_goal)
        robot_renderer.write_joint_positions(joint_pos)
        if step == steps - 1:

            color = [0, 255, 0] if converged else [255, 0, 0]
            rr.log(f"points", rr.Points3D(goal, colors=[0, 255, 0], radii=.01), rr.CoordinateFrame("base"))

    
    
    