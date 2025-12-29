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

    def __init__(self, max_step_size: float):
        self.max_step_size = max_step_size


    def plan(self, kinematics : RobotKinematics, start_pos: np.ndarray, end_pos: np.ndarray
            ) -> Optional[List[np.ndarray]]:
        """Plan a path from start to end position."""
        distance = np.linalg.norm(start_pos - end_pos)
        steps = int(np.ceil(distance/ self.max_step_size))
        step_size = distance / steps
        step = ((end_pos - start_pos) / distance)  * step_size
        joint_pos = []
        for x in range(steps):
            
            (_, joints) = kinematics.inverse_kinematics(start_pos + (x + 1)  * (step))
            joint_pos.append(joints)
        
        return joint_pos