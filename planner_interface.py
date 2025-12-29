from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class PathPlanner(ABC):
    """Abstract base class for path planners.

    Planners return paths as lists of joint configurations.
    """

    def _save_state(self, kinematics) -> List[float]:
        """Save current joint state."""
        return [kinematics.robot.get_joint(name)
                for name in kinematics.robot.joint_names()]

    def _restore_state(self, kinematics, saved: List[float]):
        """Restore joint state."""
        for idx, name in enumerate(kinematics.robot.joint_names()):
            kinematics.robot.set_joint(name, saved[idx])
        kinematics.robot.update_kinematics()

    def generate_path(self, kinematics, start_pos: np.ndarray, end_pos: np.ndarray
                     ) -> Optional[List[np.ndarray]]:
        """Plan a path, preserving robot state.

        This is the main entry point. Saves/restores robot state around planning.
        """
        saved = self._save_state(kinematics)
        try:
            return self.plan(kinematics, start_pos, end_pos)
        finally:
            self._restore_state(kinematics, saved)

    @abstractmethod
    def plan(self, kinematics, start_pos: np.ndarray, end_pos: np.ndarray
            ) -> Optional[List[np.ndarray]]:
        """Plan a path from start to end position (Cartesian XYZ).

        Subclasses implement this. May modify robot state freely.

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
        saved = self._save_state(kinematics)
        try:
            start_pos = kinematics.forward_kinematics(start_joints)
            end_pos = kinematics.forward_kinematics(end_joints)
            return self.plan(kinematics, start_pos, end_pos)
        finally:
            self._restore_state(kinematics, saved)

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
