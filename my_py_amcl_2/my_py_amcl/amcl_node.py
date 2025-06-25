import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
from scipy.spatial.transform import Rotation as R
import heapq
from enum import Enum
import math

from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseArray, TransformStamped, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, TransformListener, Buffer

class State(Enum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    AVOIDING_OBSTACLE = 3

class AmclNode(Node):
    def __init__(self):
        super().__init__('my_py_amcl')

        # --- Parameters ---
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('initial_pose_topic', 'initialpose')
        self.declare_parameter('laser_max_range', 3.5)
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('obstacle_detection_distance', 0.3)
        self.declare_parameter('obstacle_avoidance_turn_speed', 0.5)

        # --- Parameters to set ---
        self.declare_parameter('num_particles', 1000)
        self.declare_parameter('alpha1', 0.02)
        self.declare_parameter('alpha2', 0.02)
        self.declare_parameter('alpha3', 0.02)
        self.declare_parameter('alpha4', 0.02)
        self.declare_parameter('z_hit', 0.5)
        self.declare_parameter('z_rand', 0.01)
        self.declare_parameter('lookahead_distance', 0.3)
        self.declare_parameter('linear_velocity', 0.3)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('path_pruning_distance', 0.4)
        self.declare_parameter('safety_margin_cells', 3)

        
        
        self.num_particles = self.get_parameter('num_particles').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.laser_max_range = self.get_parameter('laser_max_range').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value
        self.alphas = np.array([
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ])
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_pruning_distance = self.get_parameter('path_pruning_distance').value
        self.safety_margin_cells = self.get_parameter('safety_margin_cells').value
        self.obstacle_detection_distance = self.get_parameter('obstacle_detection_distance').value
        self.obstacle_avoidance_turn_speed = self.get_parameter('obstacle_avoidance_turn_speed').value

        # --- State ---
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.map_data = None
        self.latest_scan = None
        self.initial_pose_received = False
        self.map_received = False
        self.last_odom_pose = None
        self.state = State.IDLE
        self.current_path = None
        self.goal_pose = None
        self.inflated_grid = None
        self.obstacle_avoidance_start_yaw = None
        self.obstacle_avoidance_last_yaw = None
        self.obstacle_avoidance_cumulative_angle = 0.0
        self.obstacle_avoidance_active = False
        
        # --- ROS 2 Interfaces ---
        map_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        scan_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        
        self.map_sub = self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, scan_qos)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.get_parameter('initial_pose_topic').value, self.initial_pose_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)
        
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'amcl_pose', 10)
        self.particle_pub = self.create_publisher(MarkerArray, 'particle_cloud', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('MyPyAMCL node initialized.')

    def map_callback(self, msg):
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            self.grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
            self.inflate_map()
            self.get_logger().info('Map and inflated map processed.')

    def inflate_map(self):
        """Inflate obstacles in the map for path planning safety"""
        h, w = self.grid.shape
        self.inflated_grid = self.grid.copy()
        
        # Create a kernel for inflation
        margin = self.safety_margin_cells
        for i in range(h):
            for j in range(w):
                if self.grid[i, j] == 100:  # Obstacle
                    # Inflate around this obstacle
                    for di in range(-margin, margin + 1):
                        for dj in range(-margin, margin + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                if self.inflated_grid[ni, nj] == 0:  # Free space
                                    self.inflated_grid[ni, nj] = 50  # Mark as inflated

    def scan_callback(self, msg):
        self.latest_scan = msg

    def goal_callback(self, msg):
        if self.map_data is None:
            self.get_logger().warn("Goal received, but map is not available yet. Ignoring goal.")
            return

        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Goal received in frame '{msg.header.frame_id}', but expected '{self.map_frame_id}'. Ignoring.")
            return
            
        self.goal_pose = msg.pose
        self.get_logger().info(f"New goal received: ({self.goal_pose.position.x:.2f}, {self.goal_pose.position.y:.2f}). State -> PLANNING")
        self.state = State.PLANNING
        self.current_path = None

    def initial_pose_callback(self, msg):
        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Initial pose frame is '{msg.header.frame_id}' but expected '{self.map_frame_id}'. Ignoring.")
            return
        self.get_logger().info('Initial pose received.')
        self.initialize_particles(msg.pose.pose)
        self.initial_pose_received = True
        self.last_odom_pose = None # Reset odom tracking

    def initialize_particles(self, initial_pose):
        """Initialize particles around the given initial pose with random variations"""
        # Extract position and orientation from initial pose
        x = initial_pose.position.x
        y = initial_pose.position.y
        q = initial_pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        
        # Add random noise around the initial pose
        pos_std = 0.5  # Standard deviation for position (meters)
        yaw_std = 0.3  # Standard deviation for orientation (radians)
        
        for i in range(self.num_particles):
            self.particles[i, 0] = x + np.random.normal(0, pos_std)
            self.particles[i, 1] = y + np.random.normal(0, pos_std)
            self.particles[i, 2] = yaw + np.random.normal(0, yaw_std)
            
            # Normalize angle to [-pi, pi]
            self.particles[i, 2] = np.arctan2(np.sin(self.particles[i, 2]), np.cos(self.particles[i, 2]))
        
        # Reset weights to uniform
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.publish_particles()

    def initialize_particles_randomly(self):
        """Initialize particles randomly across free space in the map"""
        if self.map_data is None:
            return
            
        free_cells = []
        h, w = self.grid.shape
        
        # Find all free cells in the map
        for i in range(h):
            for j in range(w):
                if self.grid[i, j] == 0:  # Free space
                    wx, wy = self.grid_to_world(j, i)
                    free_cells.append((wx, wy))
        
        if len(free_cells) == 0:
            self.get_logger().warn("No free cells found in map for random initialization")
            return
            
        # Randomly sample particles from free cells
        for i in range(self.num_particles):
            if len(free_cells) > 0:
                wx, wy = free_cells[np.random.randint(len(free_cells))]
                self.particles[i, 0] = wx
                self.particles[i, 1] = wy
                self.particles[i, 2] = np.random.uniform(-np.pi, np.pi)
            
        # Reset weights to uniform
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.publish_particles()

    def timer_callback(self):
        """Main timer callback implementing the state machine"""
        if not self.map_received:
            return

        # --- Localization (always running) ---
        if self.latest_scan is None:
            return
            
        if not self.initial_pose_received:
            self.initialize_particles_randomly()
            self.initial_pose_received = True
            return

        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            if self.state in [State.NAVIGATING, State.AVOIDING_OBSTACLE]:
                self.stop_robot()
            return

        # Update particle filter
        self.motion_model(current_odom_tf)
        self.measurement_model()
        self.resample()
        estimated_pose = self.estimate_pose()

        # State machine for navigation
        if self.state == State.PLANNING:
            self.plan_path(estimated_pose)
        elif self.state == State.NAVIGATING:
            self.navigate_to_goal(estimated_pose)
        elif self.state == State.AVOIDING_OBSTACLE:
            self.avoid_obstacle(estimated_pose)

        # Always publish localization results
        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)

    def plan_path(self, current_pose):
        """Simple A* path planning"""
        if self.goal_pose is None:
            self.state = State.IDLE
            return
            
        start_gx, start_gy = self.world_to_grid(current_pose.position.x, current_pose.position.y)
        goal_gx, goal_gy = self.world_to_grid(self.goal_pose.position.x, self.goal_pose.position.y)
        
        path = self.astar_planning(start_gx, start_gy, goal_gx, goal_gy)
        
        if path:
            self.current_path = path
            self.state = State.NAVIGATING
            self.get_logger().info("Path planned successfully -> NAVIGATING")
            
            # Publish path for visualization
            path_msg = Path()
            for gx, gy in path:
                wx, wy = self.grid_to_world(gx, gy)
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = wx
                pose_stamped.pose.position.y = wy
                path_msg.poses.append(pose_stamped)
            self.publish_path(path_msg)
        else:
            self.get_logger().warn("Failed to plan path -> IDLE")
            self.state = State.IDLE

    def astar_planning(self, start_gx, start_gy, goal_gx, goal_gy):
        """Simple A* implementation"""
        h, w = self.inflated_grid.shape
        
        if not (0 <= start_gx < w and 0 <= start_gy < h and 
                0 <= goal_gx < w and 0 <= goal_gy < h):
            return None
            
        if self.inflated_grid[start_gy, start_gx] != 0 or self.inflated_grid[goal_gy, goal_gx] != 0:
            return None
            
        # A* implementation
        open_set = [(0, start_gx, start_gy)]
        came_from = {}
        g_score = {(start_gx, start_gy): 0}
        f_score = {(start_gx, start_gy): self.heuristic(start_gx, start_gy, goal_gx, goal_gy)}
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while open_set:
            current_f, current_x, current_y = heapq.heappop(open_set)
            
            if current_x == goal_gx and current_y == goal_gy:
                # Reconstruct path
                path = []
                while (current_x, current_y) in came_from:
                    path.append((current_x, current_y))
                    current_x, current_y = came_from[(current_x, current_y)]
                path.append((start_gx, start_gy))
                return path[::-1]
                
            for dx, dy in directions:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                
                if not (0 <= neighbor_x < w and 0 <= neighbor_y < h):
                    continue
                    
                if self.inflated_grid[neighbor_y, neighbor_x] != 0:
                    continue
                    
                tentative_g_score = g_score[(current_x, current_y)] + np.sqrt(dx*dx + dy*dy)
                
                if (neighbor_x, neighbor_y) not in g_score or tentative_g_score < g_score[(neighbor_x, neighbor_y)]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[(neighbor_x, neighbor_y)] = tentative_g_score
                    f_score[(neighbor_x, neighbor_y)] = tentative_g_score + self.heuristic(neighbor_x, neighbor_y, goal_gx, goal_gy)
                    heapq.heappush(open_set, (f_score[(neighbor_x, neighbor_y)], neighbor_x, neighbor_y))
        
        return None

    def heuristic(self, x1, y1, x2, y2):
        """Euclidean distance heuristic for A*"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def navigate_to_goal(self, current_pose):
        """Pure pursuit navigation"""
        if self.current_path is None or len(self.current_path) == 0:
            self.state = State.IDLE
            return
            
        # Check if goal is reached
        goal_dist = np.sqrt((current_pose.position.x - self.goal_pose.position.x)**2 + 
                           (current_pose.position.y - self.goal_pose.position.y)**2)
        
        if goal_dist < self.goal_tolerance:
            self.stop_robot()
            self.state = State.IDLE
            self.get_logger().info("Goal reached -> IDLE")
            return
            
        # Check for obstacles
        if self.detect_obstacle():
            self.state = State.AVOIDING_OBSTACLE
            self.obstacle_avoidance_start_yaw = self.get_current_yaw(current_pose)
            self.obstacle_avoidance_cumulative_angle = 0.0
            self.get_logger().info("Obstacle detected -> AVOIDING_OBSTACLE")
            return
            
        # Pure pursuit control
        target_point = self.get_lookahead_point(current_pose)
        if target_point is None:
            self.stop_robot()
            return
            
        cmd_vel = self.pure_pursuit_control(current_pose, target_point)
        self.cmd_vel_pub.publish(cmd_vel)

    def avoid_obstacle(self, current_pose):
        """Simple obstacle avoidance by turning"""
        current_yaw = self.get_current_yaw(current_pose)
        
        if not self.detect_obstacle():
            # No more obstacle, check if we've turned enough
            if abs(self.obstacle_avoidance_cumulative_angle) > np.pi/4:  # 45 degrees
                self.state = State.NAVIGATING
                self.get_logger().info("Obstacle cleared -> NAVIGATING")
                return
        
        # Keep turning
        cmd_vel = Twist()
        cmd_vel.angular.z = self.obstacle_avoidance_turn_speed
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Track cumulative angle
        if self.obstacle_avoidance_last_yaw is not None:
            angle_diff = current_yaw - self.obstacle_avoidance_last_yaw
            # Normalize angle difference
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            self.obstacle_avoidance_cumulative_angle += angle_diff
            
        self.obstacle_avoidance_last_yaw = current_yaw

    def detect_obstacle(self):
        """Simple obstacle detection using laser scan"""
        if self.latest_scan is None:
            return False
            
        # Check front sector for obstacles
        ranges = np.array(self.latest_scan.ranges)
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        
        # Check forward 90 degrees (-45 to +45 degrees)
        front_start_idx = int((-np.pi/4 - angle_min) / angle_increment)
        front_end_idx = int((np.pi/4 - angle_min) / angle_increment)
        
        front_start_idx = max(0, front_start_idx)
        front_end_idx = min(len(ranges), front_end_idx)
        
        if front_start_idx >= front_end_idx:
            return False
            
        front_ranges = ranges[front_start_idx:front_end_idx]
        valid_ranges = front_ranges[np.isfinite(front_ranges)]
        
        if len(valid_ranges) == 0:
            return False
            
        min_distance = np.min(valid_ranges)
        return min_distance < self.obstacle_detection_distance

    def get_current_yaw(self, pose):
        """Extract yaw from pose"""
        q = pose.orientation
        return R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]

    def get_lookahead_point(self, current_pose):
        """Get lookahead point for pure pursuit"""
        if self.current_path is None:
            return None
            
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (gx, gy) in enumerate(self.current_path):
            wx, wy = self.grid_to_world(gx, gy)
            dist = np.sqrt((wx - current_x)**2 + (wy - current_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # Find lookahead point
        for i in range(closest_idx, len(self.current_path)):
            gx, gy = self.current_path[i]
            wx, wy = self.grid_to_world(gx, gy)
            dist = np.sqrt((wx - current_x)**2 + (wy - current_y)**2)
            
            if dist >= self.lookahead_distance:
                return (wx, wy)
                
        # If no point found, return last point
        if len(self.current_path) > 0:
            gx, gy = self.current_path[-1]
            return self.grid_to_world(gx, gy)
            
        return None

    def pure_pursuit_control(self, current_pose, target_point):
        """Pure pursuit control algorithm"""
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        current_yaw = self.get_current_yaw(current_pose)
        
        target_x, target_y = target_point
        
        # Calculate angle to target
        alpha = np.arctan2(target_y - current_y, target_x - current_x) - current_yaw
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi 
        
        # Normalize angle difference
        omega = np.arctan(0.178) * alpha
        
        cmd_vel = Twist()
        cmd_vel.linear.x = self.linear_velocity
        cmd_vel.angular.z = omega  # Simple proportional control
        
        return cmd_vel

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

    def get_odom_transform(self):
        try:
            return self.tf_buffer.lookup_transform(self.odom_frame_id, self.base_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'Could not get transform from {self.odom_frame_id} to {self.base_frame_id}. Skipping update. Error: {e}', throttle_duration_sec=2.0)
            return None

    def motion_model(self, current_odom_tf):
        """Implement motion model to update particles based on odometry"""
        if self.last_odom_pose is None:
            self.last_odom_pose = current_odom_tf.transform
            return
            
        current_odom_pose = current_odom_tf.transform
        
        # Calculate odometry motion
        last_x = self.last_odom_pose.translation.x
        last_y = self.last_odom_pose.translation.y
        last_q = self.last_odom_pose.rotation
        last_yaw = R.from_quat([last_q.x, last_q.y, last_q.z, last_q.w]).as_euler('xyz')[2]
        
        curr_x = current_odom_pose.translation.x
        curr_y = current_odom_pose.translation.y
        curr_q = current_odom_pose.rotation
        curr_yaw = R.from_quat([curr_q.x, curr_q.y, curr_q.z, curr_q.w]).as_euler('xyz')[2]
        
        # Calculate motion in odometry frame
        delta_x = curr_x - last_x
        delta_y = curr_y - last_y
        delta_yaw = curr_yaw - last_yaw
        
        # Normalize angle
        delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
        
        # Convert to motion parameters
        delta_trans = np.sqrt(delta_x**2 + delta_y**2)
        delta_rot1 = np.arctan2(delta_y, delta_x) - last_yaw
        delta_rot2 = delta_yaw - delta_rot1
        
        # Normalize angles
        delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
        delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))
        
        # Apply motion model to each particle
        for i in range(self.num_particles):
            # Add noise based on motion
            delta_rot1_hat = delta_rot1 - np.random.normal(0, self.alphas[0] * abs(delta_rot1) + self.alphas[1] * delta_trans)
            delta_trans_hat = delta_trans - np.random.normal(0, self.alphas[2] * delta_trans + self.alphas[3] * (abs(delta_rot1) + abs(delta_rot2)))
            delta_rot2_hat = delta_rot2 - np.random.normal(0, self.alphas[0] * abs(delta_rot2) + self.alphas[1] * delta_trans)
            
            # Update particle pose
            self.particles[i, 0] += delta_trans_hat * np.cos(self.particles[i, 2] + delta_rot1_hat)
            self.particles[i, 1] += delta_trans_hat * np.sin(self.particles[i, 2] + delta_rot1_hat)
            self.particles[i, 2] += delta_rot1_hat + delta_rot2_hat
            
            # Normalize angle
            self.particles[i, 2] = np.arctan2(np.sin(self.particles[i, 2]), np.cos(self.particles[i, 2]))
        
        self.last_odom_pose = current_odom_pose

    def measurement_model(self):
        """Implement measurement model to update particle weights based on laser scan"""
        if self.latest_scan is None:
            return
            
        map_res = self.map_data.info.resolution
        map_origin = self.map_data.info.origin.position
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        map_img = np.array(self.map_data.data).reshape((map_h, map_w))

        scan_ranges = np.array(self.latest_scan.ranges)
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        
        # Update weight for each particle
        for i in range(self.num_particles):
            particle_x, particle_y, particle_yaw = self.particles[i]
            
            # Calculate probability for this particle
            prob = 1.0
            
            # Sample some laser rays (not all for efficiency)
            step = max(1, len(scan_ranges) // 50)  # Sample ~50 rays
            
            for j in range(0, len(scan_ranges), step):
                if not np.isfinite(scan_ranges[j]):
                    continue
                    
                # Ray angle in world frame
                ray_angle = particle_yaw + angle_min + j * angle_increment
                
                # Expected range based on map
                expected_range = self.ray_casting(particle_x, particle_y, ray_angle, map_img, map_res, map_origin, map_w, map_h)
                
                if expected_range is None:
                    continue
                    
                # Calculate probability using beam model
                observed_range = min(scan_ranges[j], self.laser_max_range)
                
                # Simple beam model: Gaussian + uniform
                if expected_range < self.laser_max_range:
                    # Hit probability
                    sigma = 0.2  # Standard deviation
                    p_hit = np.exp(-0.5 * ((observed_range - expected_range) / sigma) ** 2)
                    p_hit /= (sigma * np.sqrt(2 * np.pi))
                else:
                    p_hit = 0.0
                
                # Random probability
                p_rand = 1.0 / self.laser_max_range
                
                # Combined probability
                p_total = self.z_hit * p_hit + self.z_rand * p_rand
                prob *= max(p_total, 1e-6)  # Avoid zero probability
            
            self.weights[i] = prob
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def ray_casting(self, start_x, start_y, angle, map_img, map_res, map_origin, map_w, map_h):
        """Cast a ray from start position at given angle and return distance to obstacle"""
        # Ray direction
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Step size for ray casting
        step_size = map_res / 2.0
        max_steps = int(self.laser_max_range / step_size)
        
        for step in range(max_steps):
            # Current position along ray
            x = start_x + step * step_size * dx
            y = start_y + step * step_size * dy
            
            # Convert to grid coordinates
            gx = int((x - map_origin.x) / map_res)
            gy = int((y - map_origin.y) / map_res)
            
            # Check bounds
            if gx < 0 or gx >= map_w or gy < 0 or gy >= map_h:
                return step * step_size
                
            # Check if hit obstacle
            if map_img[gy, gx] > 50:  # Occupied or unknown
                return step * step_size
        
        return self.laser_max_range

    def resample(self):
        """Resample particles based on weights using systematic resampling"""
        # Calculate effective sample size
        eff_sample_size = 1.0 / np.sum(self.weights ** 2)
        
        # Only resample if effective sample size is too low
        if eff_sample_size < self.num_particles / 2.0:
            # Systematic resampling
            new_particles = np.zeros_like(self.particles)
            new_weights = np.ones(self.num_particles) / self.num_particles
            
            # Create cumulative distribution
            cumulative_weights = np.cumsum(self.weights)
            
            # Generate systematic samples
            step = 1.0 / self.num_particles
            start = np.random.uniform(0, step)
            
            for i in range(self.num_particles):
                target = start + i * step
                
                # Find particle to resample
                idx = np.searchsorted(cumulative_weights, target)
                idx = min(idx, self.num_particles - 1)
                
                new_particles[i] = self.particles[idx].copy()
            
            self.particles = new_particles
            self.weights = new_weights

    def estimate_pose(self):
        """Estimate pose from particles and weights using weighted average"""
        # Weighted average of particle positions
        x = np.sum(self.particles[:, 0] * self.weights)
        y = np.sum(self.particles[:, 1] * self.weights)
        
        # For angle, use circular statistics
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        yaw = np.arctan2(sin_sum, cos_sum)
        
        # Create pose message
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        
        # Convert yaw to quaternion
        q = R.from_euler('z', yaw).as_quat()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        
        return pose

    def publish_pose(self, estimated_pose):
        p = PoseWithCovarianceStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = self.map_frame_id
        p.pose.pose = estimated_pose
        self.pose_pub.publish(p)

    def publish_particles(self):
        ma = MarkerArray()
        for i, p in enumerate(self.particles):
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            q = R.from_euler('z', p[2]).as_quat()
            marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 0.5
            marker.color.r = 1.0
            ma.markers.append(marker)
        self.particle_pub.publish(ma)

    def publish_transform(self, estimated_pose, odom_tf):
        """Publish transform from map to odom frame"""
        map_to_base_mat = self.pose_to_matrix(estimated_pose)
        odom_to_base_mat = self.transform_to_matrix(odom_tf.transform)
        map_to_odom_mat = np.dot(map_to_base_mat, np.linalg.inv(odom_to_base_mat))
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame_id
        t.child_frame_id = self.odom_frame_id
        
        # Extract translation
        t.transform.translation.x = map_to_odom_mat[0, 3]
        t.transform.translation.y = map_to_odom_mat[1, 3]
        t.transform.translation.z = map_to_odom_mat[2, 3]
        
        # Extract rotation
        rotation_matrix = map_to_odom_mat[:3, :3]
        r = R.from_matrix(rotation_matrix)
        q = r.as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def pose_to_matrix(self, pose):
        q = pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return mat

    def transform_to_matrix(self, transform):
        q = transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        t = transform.translation
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        gy = int((wy - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.map_data.info.resolution + self.map_data.info.origin.position.x
        wy = gy * self.map_data.info.resolution + self.map_data.info.origin.position.y
        return (wx, wy)
    
    def publish_path(self, path_msg):
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame_id
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


        # def measurement_model(self):
    #     """Implement measurement model to update particle weights based on laser scan"""
    #     if self.latest_scan is None:
    #         return
            
    #     map_res = self.map_data.info.resolution
    #     map_origin = self.map_data.info.origin.position
    #     map_w = self.map_data.info.width
    #     map_h = self.map_data.info.height
    #     map_img = np.array(self.map_data.data).reshape((map_h, map_w))

    #     scan_ranges = np.array(self.latest_scan.ranges)
    #     angle_min = self.latest_scan.angle_min
    #     angle_increment = self.latest_scan.angle_increment
        
    #     # Update weight for each particle
    #     for i in range(self.num_particles):
    #         particle_x, particle_y, particle_yaw = self.particles[i]
            
    #         # Verificar si la partícula está en una posición válida
    #         gx, gy = self.world_to_grid(particle_x, particle_y)
    #         if (gx < 0 or gx >= map_w or gy < 0 or gy >= map_h or 
    #             map_img[gy, gx] > 50):
    #             self.weights[i] = 1e-10  # Peso muy bajo para partículas en obstáculos
    #             continue
            
    #         # Calculate probability for this particle
    #         log_prob = 0.0  # Usar log-probabilidad para evitar underflow
            
    #         # CAMBIO: Usar más rayos para mejor precisión
    #         step = max(1, len(scan_ranges) // 100)  # Aumentado de 50 a 100 rayos
            
    #         valid_rays = 0
    #         for j in range(0, len(scan_ranges), step):
    #             if not np.isfinite(scan_ranges[j]) or scan_ranges[j] <= 0:
    #                 continue
                    
    #             # Ray angle in world frame
    #             ray_angle = particle_yaw + angle_min + j * angle_increment
                
    #             # Expected range based on map
    #             expected_range = self.ray_casting(particle_x, particle_y, ray_angle, 
    #                                             map_img, map_res, map_origin, map_w, map_h)
                
    #             if expected_range is None:
    #                 continue
                    
    #             # Calculate probability using beam model
    #             observed_range = min(scan_ranges[j], self.laser_max_range)
                
    #             # CAMBIO: Modelo de haz mejorado
    #             if expected_range < self.laser_max_range:
    #                 # Hit probability con distribución más ajustada
    #                 sigma = 0.1  # Reducido de 0.2 a 0.1 para ser más selectivo
    #                 diff = abs(observed_range - expected_range)
    #                 p_hit = np.exp(-0.5 * (diff / sigma) ** 2)
    #             else:
    #                 p_hit = 0.1 if observed_range >= self.laser_max_range * 0.9 else 0.01
                
    #             # Random probability
    #             p_rand = 0.1 / self.laser_max_range
                
    #             # Combined probability
    #             p_total = self.z_hit * p_hit + self.z_rand * p_rand
    #             log_prob += np.log(max(p_total, 1e-10))  # Log para evitar underflow
    #             valid_rays += 1
            
    #         # Convertir de log-probabilidad a probabilidad
    #         if valid_rays > 0:
    #             self.weights[i] = np.exp(log_prob / valid_rays)  # Normalizar por número de rayos
    #         else:
    #             self.weights[i] = 1e-10
        
    #     # Normalize weights
    #     weight_sum = np.sum(self.weights)
    #     if weight_sum > 0:
    #         self.weights /= weight_sum
    #     else:
    #         self.weights = np.ones(self.num_particles) / self.num_particles
