import placo
import numpy as np

class RobotKinematics:
    def __init__(self, urdf_path, end_frame, collisions=False):
        self.end_frame = end_frame
        flags = 0
        if collisions:
            flags = placo.Flags.ignore_collisions
        self.robot = placo.RobotWrapper(urdf_path, flags)

        # Initializing the kinematics solver
        # print("Initializing Solver and Palco")
        self.solver = placo.KinematicsSolver(self.robot)
        # The floating base can't move
        self.solver.mask_fbase(True)

        # Adding a frame task
        self.effector_task = self.solver.add_position_task(self.end_frame, np.zeros(3))
    # returns array of final position of end frame 
    def forward_kinematics(self, joint_positions):
        for (idx, joint_name) in enumerate(self.robot.joint_names()):
            self.robot.set_joint(joint_name, joint_positions[idx])
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.end_frame)[:3, 3]
    def get_ee_pos(self):
        return self.robot.get_T_world_frame(self.end_frame)[:3, 3]
    # returns tuple (converged, joint positions)
    def inverse_kinematics(self, end_position):
        self.effector_task.target_world = end_position
        self.robot.update_kinematics()
        # Solving the IK
        for _ in range(25):
            self.solver.solve(True)
            self.robot.update_kinematics()
        position = self.robot.get_T_world_frame(self.end_frame)[:3, 3]
        residual = np.linalg.norm(end_position - position)
        converged = residual < 0.01
        joint_positions = []
        for (name) in (self.robot.joint_names()):
            joint_positions.append(self.robot.get_joint(name))
        return (converged, joint_positions)