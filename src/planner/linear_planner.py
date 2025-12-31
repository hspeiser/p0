import numpy as np
import heapq
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from .planner_interface import PathPlanner
from kinematics import RobotKinematics

class LinearPlanner(PathPlanner):
    """Probabilistic Roadmap planner.

    Precomputes a roadmap of collision-free configurations,
    then uses A* to find paths through the roadmap.
    """

    def __init__(self, max_step_size: float, max_reroutes: int = 3):
        self.max_step_size = max_step_size
        self.max_reroutes = max_reroutes


    def plan(self, kinematics : RobotKinematics, start_pos: np.ndarray, end_pos: np.ndarray
            ) -> Tuple[bool, Optional[List[np.ndarray]]]:
        """Plan a path from start to end position."""
        goal = end_pos
        start = start_pos
        joint_pos = []
        x = 0
        collided = False
        reroutes = 0
        safe_goal = np.array([0.20, 0, (start_pos[2] + end_pos[2]) / 2])
        initial_joints = kinematics.get_joints()
        while True:
            distance = np.linalg.norm(start - goal)
            steps = int(np.ceil(distance/ self.max_step_size))
            step_size = distance / steps
            step = ((goal - start) / distance)  * step_size
            (_, joints) = kinematics.inverse_kinematics(start + (x + 1)  * (step))
            if joints  == None:
                print("Failed QP")
                return (False, None)
            joint_pos.append(joints)
            if len(kinematics.check_collisions()) != 0:
                print(f" collisions found, rerouting")
                reroutes += 1
                failed_path = list(joint_pos)
                if reroutes > self.max_reroutes:
                    print("Exceeded reroute limit; marking as failure")
                    return (False, failed_path if failed_path else None)
                kinematics.forward_kinematics(initial_joints)
                joint_pos = []
                start = start_pos
                collided = True
                goal = safe_goal
                x = 0
            if x == steps:
                if collided:
                    collided = False
                    x = 0 
                    goal = end_pos
                    start = kinematics.get_ee_pos()
                    print("avoided collision, now going to end position")
                    continue
                print("finsihed safely")
                return (True, joint_pos)
            x += 1
        return (True, joint_pos)