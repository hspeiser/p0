import placo
import numpy as np
import json
from pathlib import Path


class RobotKinematics:
    def __init__(self, urdf_path, end_frame, avoid_collisions=True):
        self.end_frame = end_frame
        self.urdf_path = Path(urdf_path)

        # Load robot - don't ignore collisions if we want to avoid them
        flags = 0 if avoid_collisions else placo.Flags.ignore_collisions
        self.robot = placo.RobotWrapper(str(self.urdf_path), flags)

        # Initializing the kinematics solver
        self.solver = placo.KinematicsSolver(self.robot)
        # The floating base can't move
        self.solver.mask_fbase(True)

        # Load collision pairs and add avoidance constraint
        if avoid_collisions:
            self._load_collision_pairs()
            # Add self-collision avoidance constraint to the solver
            self.collision_constraint = self.solver.add_avoid_self_collisions_constraint()
            self.collision_constraint.configure("collision_avoidance", "hard", 1.0)

        # Adding a frame task
        self.effector_task = self.solver.add_position_task(self.end_frame, np.zeros(3))
        self.effector_task.configure("effector_task", "soft", 1e6)

    def _load_collision_pairs(self):
        """Load collision pairs from collisions.json next to the URDF."""
        # Look for collisions.json in the URDF directory
        if self.urdf_path.is_dir():
            collisions_path = self.urdf_path / "collisions.json"
        else:
            collisions_path = self.urdf_path.parent / "collisions.json"

        if collisions_path.exists():
            # load_collision_pairs expects a file path string, not a list
            self.robot.load_collision_pairs(str(collisions_path))
            # Count pairs for logging
            with open(collisions_path, 'r') as f:
                pairs = json.load(f)
            print(f"Loaded {len(pairs)} collision pairs from {collisions_path}")
        else:
            print(f"No collisions.json found at {collisions_path}, using default collision pairs")

    def check_collisions(self):
        """Return list of current self-collisions."""
        self.robot.update_kinematics()
        return self.robot.self_collisions()

    def get_distances(self):
        """Return minimum distances between collision pairs."""
        self.robot.update_kinematics()
        return self.robot.distances()

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
        for name in self.robot.joint_names():
            joint_positions.append(self.robot.get_joint(name))
        return (converged, joint_positions)
