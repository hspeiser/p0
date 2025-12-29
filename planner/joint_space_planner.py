import numpy as np
from planner_interface import PathPlanner
from typing import List, Optional


class JointSpacePlanner(PathPlanner):
    def __init__(self, num_steps: int = 20):
        self.num_steps = num_steps
        self.last_path = None

    def plan(self, kinematics, start_pos: np.ndarray, end_pos: np.ndarray
            ) -> Optional[List[np.ndarray]]:
        # Get current joints as start
        start_joints = np.array([
            kinematics.robot.get_joint(name)
            for name in kinematics.robot.joint_names()
        ])

        # Get goal joints via IK
        success, end_joints = kinematics.inverse_kinematics(end_pos)
        if not success:
            return None

        return self.plan_joints(kinematics, start_joints, np.array(end_joints))

    def plan_joints(self, kinematics, start_joints: np.ndarray, end_joints: np.ndarray
                   ) -> Optional[List[np.ndarray]]:
        # Linear interpolation in joint space
        path = []
        for i in range(self.num_steps + 1):
            t = i / self.num_steps
            joints = start_joints + t * (end_joints - start_joints)
            path.append(joints)

        # Verify path is collision-free
        for joints in path:
            if not kinematics.is_collision_free(joints):
                self.last_path = None
                return None

        self.last_path = path
        return path

    def plot(self, rr=None):
        if rr and self.last_path:
            # Could visualize path waypoints here
            pass
