import placo
import numpy as np
import rerun as rr
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import time

# Initialize Rerun with remote proxy
rr.init("robot_workspace_viewer", spawn=False)
rr.connect_grpc("rerun+http://172.18.128.1:9876/proxy")

# Load robot
robot = placo.RobotWrapper("../../rerun_arm/", placo.Flags.ignore_collisions)
solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)

FRAME_NAME = "gripper_frame_link"  # Change to "jaw" if needed

def test_ik_convergence(target_pos, max_iterations=50, tolerance=2e-3):
    """Test if a position is reachable."""
    robot.state.q = np.zeros(robot.state.q.shape)
    robot.update_kinematics()
    
    task = solver.add_position_task(FRAME_NAME, np.zeros(3))
    task.target_world = target_pos
    
    for i in range(max_iterations):
        solver.solve(True)
        robot.update_kinematics()
        
        current_pos = robot.get_T_world_frame(FRAME_NAME)[:3, 3]
        error = np.linalg.norm(target_pos - current_pos)
        
        if error < tolerance:
            solver.remove_task(task)
            return True, error
    
    solver.remove_task(task)
    current_pos = robot.get_T_world_frame(FRAME_NAME)[:3, 3]
    final_error = np.linalg.norm(target_pos - current_pos)
    return False, final_error

def dense_grid_search(x_range, y_range, z_range, resolution=0.025):
    """Perform dense grid search of workspace."""
    x_vals = np.arange(x_range[0], x_range[1] + resolution, resolution)
    y_vals = np.arange(y_range[0], y_range[1] + resolution, resolution)
    z_vals = np.arange(z_range[0], z_range[1] + resolution, resolution)
    
    total = len(x_vals) * len(y_vals) * len(z_vals)
    print(f"Testing {total} points at {resolution}m resolution...")
    
    reachable_points = []
    count = 0
    start_time = time.time()
    
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            for k, z in enumerate(z_vals):
                target = np.array([x, y, z])
                converged, error = test_ik_convergence(target)
                
                if converged:
                    reachable_points.append(target)
                
                count += 1
                if count % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed
                    remaining = (total - count) / rate
                    print(f"  {count}/{total} ({100*count/total:.1f}%) - "
                          f"Reachable: {len(reachable_points)} - "
                          f"ETA: {remaining/60:.1f}min")
    
    return np.array(reachable_points)

def cluster_and_hull(points, eps=0.06, min_samples=20):
    """Cluster points and compute convex hulls."""
    if len(points) < min_samples:
        return [points], [None]
    
    print(f"\nClustering {len(points)} points...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters")
    
    clusters = []
    hulls = []
    
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        print(f"  Cluster {i}: {len(cluster_points)} points")
        clusters.append(cluster_points)
        
        if len(cluster_points) >= 4:
            try:
                hull = ConvexHull(cluster_points)
                hulls.append(hull)
                print(f"    Hull: {len(hull.vertices)} vertices, {len(hull.simplices)} faces")
            except Exception as e:
                print(f"    Hull failed: {e}")
                hulls.append(None)
        else:
            hulls.append(None)
    
    return clusters, hulls

def log_robot_to_rerun(robot, urdf_path, entity_path="world/robot", first_time=False):
    """Log robot state to Rerun using URDF."""
    robot.update_kinematics()
    
    # Log the URDF file only once using log_file_from_path
    if first_time:
        import os
        if os.path.exists(urdf_path):
            rr.log_file_from_path(urdf_path, entity_path_prefix=entity_path)
        else:
            print(f"Warning: URDF file not found at {urdf_path}")
    
    # Log each frame transform
    for frame in robot.model.frames:
        frame_name = frame.name
        T_world_frame = robot.get_T_world_frame(frame_name)
        
        position = T_world_frame[:3, 3]
        rotation_matrix = T_world_frame[:3, :3]
        
        # Convert rotation matrix to quaternion using scipy
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rotation_matrix)
        quat = r.as_quat()  # Returns [x, y, z, w]
        
        rr.log(
            f"{entity_path}/{frame_name}",
            rr.Transform3D(
                translation=position,
                quaternion=quat,
            )
        )

def log_workspace_to_rerun(reachable_points, clusters, hulls):
    """Log workspace visualization to Rerun."""
    
    # Log all reachable points
    rr.log(
        "world/workspace/reachable_points",
        rr.Points3D(
            positions=reachable_points,
            colors=[0, 255, 0, 100],  # Green, semi-transparent
            radii=0.005
        )
    )
    
    # Log each cluster with different colors
    colors_rgb = [
        [255, 100, 100],  # Red
        [100, 100, 255],  # Blue
        [255, 255, 100],  # Yellow
        [255, 100, 255],  # Magenta
        [100, 255, 255],  # Cyan
        [255, 150, 100],  # Orange
    ]
    
    for i, (cluster, hull) in enumerate(zip(clusters, hulls)):
        color = colors_rgb[i % len(colors_rgb)]
        
        # Log cluster points
        rr.log(
            f"world/workspace/cluster_{i}/points",
            rr.Points3D(
                positions=cluster,
                colors=color + [200],
                radii=0.008
            )
        )
        
        # Log convex hull as mesh
        if hull is not None:
            vertices = cluster[hull.vertices]
            
            # Convert simplices (faces) to the format Rerun expects
            # Each simplex is a triangle (3 vertex indices)
            indices = hull.simplices.flatten().tolist()
            
            rr.log(
                f"world/workspace/cluster_{i}/hull",
                rr.Mesh3D(
                    vertex_positions=cluster,
                    triangle_indices=hull.simplices,
                    vertex_colors=color + [80]  # Semi-transparent
                )
            )
            
            # Also log hull edges for better visibility
            edge_points = []
            for simplex in hull.simplices:
                for j in range(3):
                    edge_points.append(cluster[simplex[j]])
                    edge_points.append(cluster[simplex[(j + 1) % 3]])
            
            edge_points = np.array(edge_points)
            rr.log(
                f"world/workspace/cluster_{i}/hull_edges",
                rr.LineStrips3D(
                    strips=edge_points.reshape(-1, 2, 3),
                    colors=color + [255]
                )
            )

def animate_robot_reaching(robot, solver, goal_positions, urdf_path, fps=30):
    """Animate robot reaching to different positions."""
    first_iteration = True
    
    for goal_idx, goal in enumerate(goal_positions):
        print(f"\nReaching goal {goal_idx + 1}/{len(goal_positions)}: {goal}")
        
        # Reset robot
        robot.state.q = np.zeros(robot.state.q.shape)
        robot.update_kinematics()
        
        # Set goal
        task = solver.add_position_task(FRAME_NAME, np.zeros(3))
        task.target_world = goal
        
        # Solve IK iteratively and log each step
        for iteration in range(50):
            solver.solve(True)
            robot.update_kinematics()
            
            # Log current robot state (load URDF only on first iteration)
            rr.set_time_sequence("iteration", goal_idx * 50 + iteration)
            log_robot_to_rerun(robot, urdf_path, first_time=first_iteration)
            first_iteration = False
            
            # Log goal position
            rr.log(
                "world/goal",
                rr.Points3D(
                    positions=[goal],
                    colors=[255, 0, 0, 255],  # Red
                    radii=0.015
                )
            )
            
            # Log end effector trajectory
            current_pos = robot.get_T_world_frame(FRAME_NAME)[:3, 3]
            rr.log(
                "world/end_effector",
                rr.Points3D(
                    positions=[current_pos],
                    colors=[0, 255, 255, 255],  # Cyan
                    radii=0.012
                )
            )
            
            # Check convergence
            error = np.linalg.norm(current_pos - goal)
            if error < 0.002:
                print(f"  Converged at iteration {iteration}")
                break
            
            time.sleep(1.0 / fps)
        
        solver.remove_task(task)
        time.sleep(0.5)  # Pause between goals

if __name__ == "__main__":
    print("Rerun Workspace Visualization")
    print("="*70)
    
    # URDF path - adjust to your robot's URDF file
    urdf_path = "../../rerun_arm/robot.urdf"  # Change this to your actual URDF path
    
    # Search parameters
    x_range = (-0.30, 0.60)
    y_range = (-0.60, 0.60)
    z_range = (-0.10, 0.65)
    resolution = 0.04  # 4cm for faster testing (decrease for more detail)
    
    # Perform grid search
    print("\n1. Performing workspace grid search...")
    reachable = dense_grid_search(x_range, y_range, z_range, resolution)
    
    if len(reachable) == 0:
        print("No reachable points found!")
        exit(1)
    
    print(f"\nFound {len(reachable)} reachable points")
    
    # Cluster and compute hulls
    print("\n2. Clustering and computing convex hulls...")
    clusters, hulls = cluster_and_hull(reachable, eps=0.08, min_samples=15)
    
    # Log workspace to Rerun
    print("\n3. Logging workspace to Rerun...")
    rr.set_time_sequence("iteration", 0)
    log_workspace_to_rerun(reachable, clusters, hulls)
    
    # Log initial robot state with URDF
    print("\n4. Logging initial robot state with URDF...")
    robot.state.q = np.zeros(robot.state.q.shape)
    robot.update_kinematics()
    log_robot_to_rerun(robot, urdf_path, first_time=True)
    
    # Select some goals from different clusters for animation
    print("\n5. Animating robot reaching goals...")
    goal_positions = []
    for cluster in clusters[:3]:  # Use first 3 clusters
        if len(cluster) > 0:
            # Pick a random point from each cluster
            idx = np.random.randint(len(cluster))
            goal_positions.append(cluster[idx])
    
    if len(goal_positions) > 0:
        animate_robot_reaching(robot, solver, goal_positions, urdf_path, fps=20)
    
    print("\n✓ Visualization complete! Check the Rerun viewer.")
    print("  Use the timeline slider to see the robot animation.")
    print("  Toggle layers in the viewer to show/hide different elements.")