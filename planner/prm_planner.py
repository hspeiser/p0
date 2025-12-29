import numpy as np
import heapq
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from planner_interface import PathPlanner


class PRMPlanner(PathPlanner):
    """Probabilistic Roadmap planner.

    Precomputes a roadmap of collision-free configurations,
    then uses A* to find paths through the roadmap.
    """

    def __init__(self, num_samples: int = 500, k_neighbors: int = 10,
                 edge_resolution: int = 10, interpolation_steps: int = 20):
        """
        Args:
            num_samples: Number of random configurations to sample
            k_neighbors: Number of nearest neighbors to try connecting
            edge_resolution: Number of collision checks per edge
            interpolation_steps: Number of steps when interpolating between configs
        """
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.edge_resolution = edge_resolution
        self.interpolation_steps = interpolation_steps

        # Roadmap data
        self.nodes: List[np.ndarray] = []  # List of joint configurations
        self.edges: Dict[int, List[int]] = {}  # Adjacency list
        self.joint_limits: List[Tuple[float, float]] = []

        self.last_path = None
        self.roadmap_built = False

    def set_workspace(self, r_min: float = 0.15, r_max: float = 0.35,
                      theta_min: float = -np.pi/2, theta_max: float = np.pi/2,
                      phi_min: float = 0.1, phi_max: float = np.pi * 0.7,
                      z_offset: float = 0.05):
        """Set the Cartesian workspace bounds for sampling."""
        self.workspace = {
            'r_min': r_min, 'r_max': r_max,
            'theta_min': theta_min, 'theta_max': theta_max,
            'phi_min': phi_min, 'phi_max': phi_max,
            'z_offset': z_offset
        }

    def _sample_workspace_point(self) -> np.ndarray:
        """Sample a random Cartesian point in the target workspace."""
        ws = self.workspace

        # Sample in spherical coordinates (same as generate_random_goal)
        theta = np.random.uniform(ws['theta_min'], ws['theta_max'])

        # Uniform sampling on sphere surface for phi
        u = np.random.uniform(np.cos(ws['phi_max']), np.cos(ws['phi_min']))
        phi = np.arccos(u)

        # Uniform sampling in volume for r
        r = ((ws['r_max']**3 - ws['r_min']**3) * np.random.random() + ws['r_min']**3) ** (1/3)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) + ws['z_offset']

        return np.array([x, y, z])

    def build_roadmap(self, kinematics, verbose: bool = True):
        """Build the PRM roadmap by sampling Cartesian workspace and using IK."""
        if verbose:
            print("Building PRM roadmap...")

        # Initialize workspace if not set
        if not hasattr(self, 'workspace'):
            self.set_workspace()

        # Get joint limits for random config fallback
        self._extract_joint_limits(kinematics)

        # Sample valid configurations from Cartesian workspace
        if verbose:
            print(f"  Sampling {self.num_samples} configurations in workspace...")

        starting_count = len(self.nodes)
        attempts = 0
        max_attempts = self.num_samples * 20  # More attempts since IK can fail

        while len(self.nodes) - starting_count < self.num_samples and attempts < max_attempts:
            attempts += 1

            # Sample Cartesian point in workspace
            cart_point = self._sample_workspace_point()

            # Use IK to get joint configuration
            success, joints = kinematics.inverse_kinematics(cart_point)

            if success and joints is not None:
                joints = np.array(joints)
                # Check if this configuration is collision-free
                if kinematics.is_collision_free(joints):
                    # Check if we already have a very similar configuration
                    if not self._is_duplicate(joints, threshold=0.1):
                        node_idx = len(self.nodes)
                        self.nodes.append(joints)
                        self.edges[node_idx] = []

                        sampled = len(self.nodes) - starting_count
                        if verbose and sampled % 100 == 0:
                            print(f"    Found {sampled} valid configurations...")

        if verbose:
            print(f"  Sampled {len(self.nodes)} valid configurations ({attempts} attempts)")

        # Connect nearby nodes
        if verbose:
            print(f"  Connecting nodes (k={self.k_neighbors})...")

        edges_added = 0
        for i, node_i in enumerate(self.nodes):
            # Find k nearest neighbors
            distances = []
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    dist = np.linalg.norm(node_i - node_j)
                    distances.append((dist, j))

            distances.sort(key=lambda x: x[0])
            nearest = distances[:self.k_neighbors]

            # Try to connect to each neighbor
            for dist, j in nearest:
                if j not in self.edges[i]:
                    if self._is_edge_collision_free(kinematics, node_i, self.nodes[j]):
                        self.edges[i].append(j)
                        self.edges[j].append(i)
                        edges_added += 1

        self.roadmap_built = True

        if verbose:
            print(f"  Added {edges_added} edges")
            avg_degree = sum(len(e) for e in self.edges.values()) / len(self.nodes) if self.nodes else 0
            print(f"  Average node degree: {avg_degree:.1f}")
            print("Roadmap complete!")

    def _extract_joint_limits(self, kinematics):
        """Extract joint limits from the robot model."""
        # Default limits if not available
        self.joint_limits = []
        joint_names = kinematics.robot.joint_names()

        for name in joint_names:
            # Try to get limits from robot model, fallback to defaults
            try:
                # Placo might have different API - use safe defaults
                self.joint_limits.append((-1.57, 1.57))
            except:
                self.joint_limits.append((-1.57, 1.57))

    def _is_duplicate(self, joints: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if a configuration is too similar to existing nodes."""
        for node in self.nodes:
            if np.linalg.norm(joints - node) < threshold:
                return True
        return False

    def _sample_random_config(self) -> np.ndarray:
        """Sample a random configuration within joint limits."""
        config = []
        for low, high in self.joint_limits:
            config.append(np.random.uniform(low, high))
        return np.array(config)

    def _is_edge_collision_free(self, kinematics, config_a: np.ndarray,
                                 config_b: np.ndarray) -> bool:
        """Check if the edge between two configs is collision-free."""
        for i in range(self.edge_resolution + 1):
            t = i / self.edge_resolution
            config = config_a + t * (config_b - config_a)
            if not kinematics.is_collision_free(config):
                return False
        return True

    def _find_nearest_nodes(self, config: np.ndarray, k: int = 5) -> List[int]:
        """Find the k nearest nodes in the roadmap to a configuration."""
        distances = []
        for i, node in enumerate(self.nodes):
            dist = np.linalg.norm(config - node)
            distances.append((dist, i))
        distances.sort(key=lambda x: x[0])
        return [idx for _, idx in distances[:k]]

    def _astar(self, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        """A* search through the roadmap."""
        if start_idx == goal_idx:
            return [start_idx]

        # Priority queue: (f_score, node_idx)
        open_set = [(0, start_idx)]
        came_from = {}
        g_score = {start_idx: 0}

        goal_config = self.nodes[goal_idx]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.edges.get(current, []):
                tentative_g = g_score[current] + np.linalg.norm(
                    self.nodes[current] - self.nodes[neighbor]
                )

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(
                        self.nodes[neighbor] - goal_config
                    )
                    heapq.heappush(open_set, (f_score, neighbor))

        return None  # No path found

    def plan(self, kinematics, start_pos: np.ndarray, end_pos: np.ndarray
            ) -> Optional[List[np.ndarray]]:
        """Plan a path from start to end position."""
        if not self.roadmap_built:
            self.build_roadmap(kinematics)

        # Get current joint configuration as start
        start_joints = np.array([
            kinematics.robot.get_joint(name)
            for name in kinematics.robot.joint_names()
        ])

        # Get goal joints via IK
        success, end_joints = kinematics.inverse_kinematics(end_pos)
        if not success:
            return None
        end_joints = np.array(end_joints)

        return self.plan_joints(kinematics, start_joints, end_joints)

    def plan_joints(self, kinematics, start_joints: np.ndarray,
                    end_joints: np.ndarray) -> Optional[List[np.ndarray]]:
        """Plan a path between joint configurations using the roadmap."""
        if not self.roadmap_built:
            self.build_roadmap(kinematics)

        # Try direct path first (might work for nearby configs)
        if self._is_edge_collision_free(kinematics, start_joints, end_joints):
            path = self._interpolate(start_joints, end_joints)
            self.last_path = path
            return path

        # Try planning through roadmap
        path = self._plan_through_roadmap(kinematics, start_joints, end_joints)
        if path is not None:
            self.last_path = path
            return path

        self.last_path = None
        return None

    def _plan_through_roadmap(self, kinematics, start_joints: np.ndarray,
                              end_joints: np.ndarray) -> Optional[List[np.ndarray]]:
        """Try to plan a path through the roadmap."""
        # Find nearest roadmap nodes to start and goal
        start_candidates = self._find_nearest_nodes(start_joints, k=10)
        goal_candidates = self._find_nearest_nodes(end_joints, k=10)

        # Try to connect start to roadmap
        start_node = None
        for idx in start_candidates:
            if self._is_edge_collision_free(kinematics, start_joints, self.nodes[idx]):
                start_node = idx
                break

        if start_node is None:
            return None

        # Try to connect goal to roadmap
        goal_node = None
        for idx in goal_candidates:
            if self._is_edge_collision_free(kinematics, self.nodes[idx], end_joints):
                goal_node = idx
                break

        if goal_node is None:
            return None

        # Find path through roadmap
        roadmap_path = self._astar(start_node, goal_node)

        if roadmap_path is None:
            return None

        # Build full path: start -> roadmap nodes -> goal
        full_path = []

        # Start to first roadmap node
        full_path.extend(self._interpolate(start_joints, self.nodes[roadmap_path[0]])[:-1])

        # Through roadmap
        for i in range(len(roadmap_path) - 1):
            segment = self._interpolate(self.nodes[roadmap_path[i]],
                                        self.nodes[roadmap_path[i + 1]])
            full_path.extend(segment[:-1])

        # Last roadmap node to goal
        full_path.extend(self._interpolate(self.nodes[roadmap_path[-1]], end_joints))

        return full_path

    def _interpolate(self, config_a: np.ndarray, config_b: np.ndarray,
                     steps: Optional[int] = None) -> List[np.ndarray]:
        """Linearly interpolate between two configurations."""
        if steps is None:
            steps = self.interpolation_steps
        path = []
        for i in range(steps + 1):
            t = i / steps
            path.append(config_a + t * (config_b - config_a))
        return path

    def save_roadmap(self, filepath: str):
        """Save the roadmap to a JSON file."""
        data = {
            'nodes': [node.tolist() for node in self.nodes],
            'edges': {str(k): v for k, v in self.edges.items()},
            'joint_limits': self.joint_limits,
            'num_samples': self.num_samples,
            'k_neighbors': self.k_neighbors
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Roadmap saved to {filepath}")

    def load_roadmap(self, filepath: str) -> bool:
        """Load a roadmap from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.nodes = [np.array(node) for node in data['nodes']]
        self.edges = {int(k): v for k, v in data['edges'].items()}
        self.joint_limits = [tuple(lim) for lim in data['joint_limits']]
        self.num_samples = data.get('num_samples', self.num_samples)
        self.k_neighbors = data.get('k_neighbors', self.k_neighbors)
        self.roadmap_built = True

        print(f"Roadmap loaded from {filepath}: {len(self.nodes)} nodes")
        return True

    def plot(self, rr=None):
        """Visualize the roadmap."""
        if rr and self.nodes:
            # Could visualize roadmap nodes and edges
            pass
