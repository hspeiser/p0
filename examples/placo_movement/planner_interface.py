from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class PathPlanner(ABC):
    """Abstract base class for path planners.

    Planners return paths as lists of joint configurations.
    """

    @abstractmethod
    def plan(self, kinematics, start_pos: np.ndarray, end_pos: np.ndarray
            ) -> Optional[List[np.ndarray]]:
        """Plan a path from start to end position (Cartesian XYZ).

        Args:
            kinematics: RobotKinematics instance
            start_pos: Start position in Cartesian space (x, y, z)
            end_pos: End position in Cartesian space (x, y, z)

        Returns:
            List of joint configurations (numpy arrays), or None if failed.
        """
        pass

    def plan_joints(self, kinematics, start_joints: np.ndarray, end_joints: np.ndarray
                   ) -> Optional[List[np.ndarray]]:
        """Plan from joint configurations directly.

        Default converts to Cartesian and calls plan().
        Subclasses can override for more efficient joint-space planning.
        """
        start_pos = kinematics.forward_kinematics(start_joints)
        end_pos = kinematics.forward_kinematics(end_joints)
        return self.plan(kinematics, start_pos, end_pos)

    def is_path_collision_free(self, kinematics, path: List[np.ndarray]) -> bool:
        """Check if entire path is collision-free."""
        for joints in path:
            if not kinematics.is_collision_free(joints):
                return False
        return True

    def get_cartesian_path(self, kinematics, joint_path: List[np.ndarray]
                          ) -> List[np.ndarray]:
        """Convert a joint path to Cartesian positions for visualization."""
        return [kinematics.forward_kinematics(joints) for joints in joint_path]

    def plot(self, rr=None):
        """Optional: visualization/debug callback during runtime."""
        pass
