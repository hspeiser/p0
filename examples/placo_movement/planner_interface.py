from abc import ABC, abstractmethod
import placo
class PathPlanner(ABC):
    
    @abstractmethod
    def plot(self):
        """will be run to plot any data during runtime"""
        pass

    @abstractmethod
    def get_full_path(self, robot, start, end, step_size):
        """ get full path from start -> end, using robot (do not modify the robot kinematics state) with each point being at most step_size apart """
        """ array of 3D vectors """
        pass
    
    def generate_path(self, robot: placo.RobotWrapper, start, end, step_size):
        joint_pos = []
        for name in robot.joint_names():
            joint_pos.append(robot.get_joint(name))
        
        out = self.get_full_path(robot, start, end, step_size)

        for pos in joint_pos:
            robot.set_joint(pos)
        
        return out