import placo
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import time

# Load robot
robot = placo.RobotWrapper("../../rerun_arm/", placo.Flags.ignore_collisions)
solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)

def test_ik_convergence(target_pos, max_iterations=100, tolerance=1e-3):
    """
    Test if a position is reachable and return convergence info.
    
    Returns:
        (converged, final_error, iterations_used)
    """
    # Reset robot to neutral pose
    robot.state.q = np.zeros(robot.state.q.shape)
    robot.update_kinematics()
    
    # Create task
    task = solver.add_position_task("gripper_frame_link", np.zeros(3))
    task.target_world = target_pos
    
    # Solve
    for i in range(max_iterations):
        solver.solve(True)
        robot.update_kinematics()
        
        current_pos = robot.get_T_world_frame("gripper_frame_link")[:3, 3]
        error = np.linalg.norm(target_pos - current_pos)
        
        if error < tolerance:
            solver.remove_task(task)
            return True, error, i + 1
    
    solver.remove_task(task)
    current_pos = robot.get_T_world_frame("gripper_frame_link")[:3, 3]
    final_error = np.linalg.norm(target_pos - current_pos)
    return False, final_error, max_iterations

def sample_workspace_random(num_samples=500, radius_range=(0.05, 0.35)):
    """
    Sample the workspace randomly in spherical coordinates.
    
    Returns:
        reachable_points, unreachable_points
    """
    reachable = []
    unreachable = []
    
    print(f"Testing {num_samples} random points...")
    
    for i in range(num_samples):
        # Random spherical coordinates
        r = np.random.uniform(radius_range[0], radius_range[1])
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        target = np.array([x, y, z])
        converged, error, iters = test_ik_convergence(target, max_iterations=50, tolerance=2e-3)
        
        if converged:
            reachable.append(target)
        else:
            unreachable.append(target)
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{num_samples} - Reachable: {len(reachable)} ({100*len(reachable)/(i+1):.1f}%)")
    
    return np.array(reachable), np.array(unreachable)

def cluster_and_mesh_workspace(reachable_points, n_clusters=3, min_points_for_hull=10):
    """
    Cluster reachable points and create convex hull meshes for each cluster.
    
    Returns:
        clusters, hulls, cluster_labels
    """
    if len(reachable_points) < n_clusters:
        print(f"Warning: Only {len(reachable_points)} points, reducing clusters to {len(reachable_points)}")
        n_clusters = max(1, len(reachable_points))
    
    # Perform K-means clustering
    print(f"\nClustering {len(reachable_points)} points into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reachable_points)
    
    # Create convex hulls for each cluster
    hulls = []
    valid_clusters = []
    
    for i in range(n_clusters):
        cluster_points = reachable_points[labels == i]
        print(f"Cluster {i}: {len(cluster_points)} points")
        
        if len(cluster_points) >= min_points_for_hull:
            try:
                hull = ConvexHull(cluster_points)
                hulls.append(hull)
                valid_clusters.append(cluster_points)
                print(f"  ✓ Created hull with {len(hull.vertices)} vertices, {len(hull.simplices)} faces")
            except Exception as e:
                print(f"  ✗ Could not create hull: {e}")
                valid_clusters.append(cluster_points)
                hulls.append(None)
        else:
            print(f"  ✗ Not enough points for hull (need {min_points_for_hull})")
            valid_clusters.append(cluster_points)
            hulls.append(None)
    
    return valid_clusters, hulls, labels

def plot_workspace_with_hulls(reachable, unreachable, clusters, hulls, labels):
    """Plot the workspace with clustered convex hull meshes."""
    fig = plt.figure(figsize=(18, 6))
    
    # Color palette for clusters
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    # Calculate recommended sphere shell (covering 80% of points)
    distances = np.linalg.norm(reachable, axis=1)
    inner_radius = np.percentile(distances, 10)  # 10th percentile
    outer_radius = np.percentile(distances, 90)  # 90th percentile (80% coverage)
    
    # 3D view with meshes
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot unreachable points (reduced alpha)
    if len(unreachable) > 0:
        ax1.scatter(unreachable[:, 0], unreachable[:, 1], unreachable[:, 2], 
                   c='lightgray', marker='x', s=10, alpha=0.1, label='Unreachable')
    
    # Plot each cluster with its hull
    for i, (cluster_points, hull) in enumerate(zip(clusters, hulls)):
        # Plot cluster points
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   c=[colors[i]], marker='o', s=30, alpha=0.7, 
                   label=f'Cluster {i} ({len(cluster_points)} pts)')
        
        # Plot convex hull mesh
        if hull is not None:
            for simplex in hull.simplices:
                triangle = cluster_points[simplex]
                # Create a triangular face
                tri = Poly3DCollection([triangle], alpha=0.15, 
                                         facecolors=colors[i], 
                                         edgecolors=colors[i], 
                                         linewidths=0.5)
                ax1.add_collection3d(tri)
    
    # Draw recommended sphere shell (80% coverage)
    print(f"\nDrawing recommended sphere shell: r={inner_radius:.3f}m to {outer_radius:.3f}m")
    
    # Create sphere mesh
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot outer sphere (90th percentile)
    ax1.plot_surface(x_sphere * outer_radius, y_sphere * outer_radius, z_sphere * outer_radius,
                     color='blue', alpha=0.08, shade=False, linewidth=0)
    
    # Plot inner sphere (10th percentile)
    ax1.plot_surface(x_sphere * inner_radius, y_sphere * inner_radius, z_sphere * inner_radius,
                     color='blue', alpha=0.08, shade=False, linewidth=0)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Workspace with Clustered Hulls')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_box_aspect([1,1,1])
    
    # XY projection with clusters
    ax2 = fig.add_subplot(132)
    if len(unreachable) > 0:
        ax2.scatter(unreachable[:, 0], unreachable[:, 1], 
                   c='lightgray', marker='x', s=10, alpha=0.2)
    
    for i, cluster_points in enumerate(clusters):
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[colors[i]], marker='o', s=30, alpha=0.7, 
                   label=f'Cluster {i}')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection (Top View)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # XZ projection with clusters
    ax3 = fig.add_subplot(133)
    if len(unreachable) > 0:
        ax3.scatter(unreachable[:, 0], unreachable[:, 2], 
                   c='lightgray', marker='x', s=10, alpha=0.2)
    
    for i, cluster_points in enumerate(clusters):
        ax3.scatter(cluster_points[:, 0], cluster_points[:, 2], 
                   c=[colors[i]], marker='o', s=30, alpha=0.7,
                   label=f'Cluster {i}')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Projection (Side View)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('workspace_clustered.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to 'workspace_clustered.png'")
    plt.show()

def analyze_reachable_statistics(reachable):
    """Print statistics about the reachable workspace."""
    if len(reachable) == 0:
        print("No reachable points found!")
        return
    
    distances = np.linalg.norm(reachable, axis=1)
    
    print("\n" + "="*60)
    print("WORKSPACE STATISTICS")
    print("="*60)
    print(f"Reachable points: {len(reachable)}")
    print(f"\nDistance from origin:")
    print(f"  Min:    {distances.min():.4f} m")
    print(f"  Max:    {distances.max():.4f} m")
    print(f"  Mean:   {distances.mean():.4f} m")
    print(f"  Median: {np.median(distances):.4f} m")
    
    print(f"\nX range: [{reachable[:, 0].min():.4f}, {reachable[:, 0].max():.4f}]")
    print(f"Y range: [{reachable[:, 1].min():.4f}, {reachable[:, 1].max():.4f}]")
    print(f"Z range: [{reachable[:, 2].min():.4f}, {reachable[:, 2].max():.4f}]")
    
    # Calculate 80% coverage sphere
    inner_percentile = np.percentile(distances, 10)
    outer_percentile = np.percentile(distances, 90)
    
    print(f"\nRecommended safe sphere shell (80% coverage):")
    print(f"  Inner radius: {inner_percentile:.4f} m (10th percentile)")
    print(f"  Outer radius: {outer_percentile:.4f} m (90th percentile)")
    print(f"  Shell thickness: {outer_percentile - inner_percentile:.4f} m")
    print("="*60)

def generate_safe_goal(reachable_points):
    """Generate a goal that's definitely reachable based on sampled workspace."""
    if len(reachable_points) == 0:
        return None
    
    # Pick a random reachable point and add small noise
    idx = np.random.randint(len(reachable_points))
    base_point = reachable_points[idx]
    
    # Add small random offset (within 2cm)
    noise = np.random.randn(3) * 0.02
    return base_point + noise

if __name__ == "__main__":
    print("Robot Arm Workspace Analyzer with Clustering")
    print("="*60)
    
    start_time = time.time()
    
    # Random sampling
    reachable, unreachable = sample_workspace_random(num_samples=800, radius_range=(0.05, 0.4))
    
    elapsed = time.time() - start_time
    print(f"\nSampling completed in {elapsed:.1f} seconds")
    
    if len(reachable) == 0:
        print("\n⚠️  No reachable points found! Check your robot configuration.")
        exit(1)
    
    # Analyze results
    analyze_reachable_statistics(reachable)
    
    # Cluster and create meshes
    n_clusters = min(5, max(2, len(reachable) // 50))  # Auto-determine cluster count
    clusters, hulls, labels = cluster_and_mesh_workspace(reachable, n_clusters=n_clusters)
    
    # Plot workspace with hulls
    plot_workspace_with_hulls(reachable, unreachable, clusters, hulls, labels)
    
    # Test a few safe goals
    print("\n" + "="*60)
    print("TESTING SAFE GOAL GENERATION")
    print("="*60)
    for i in range(5):
        safe_goal = generate_safe_goal(reachable)
        if safe_goal is not None:
            converged, error, iters = test_ik_convergence(safe_goal, max_iterations=50, tolerance=1e-3)
            status = "✓ REACHED" if converged else "✗ FAILED"
            print(f"Goal {i+1}: {safe_goal} - {status} (error: {error:.6f})")