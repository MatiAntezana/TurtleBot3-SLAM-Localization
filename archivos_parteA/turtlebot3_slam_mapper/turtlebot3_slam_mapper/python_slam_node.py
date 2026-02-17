#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
# import tf_transformations
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import random
from copy import deepcopy

def quaternion_from_yaw(dtheta):
    qx = 0.0
    qy = 0.0
    qz = math.sin(dtheta / 2.0)
    qw = math.cos(dtheta / 2.0)
    return qx, qy, qz, qw

def quaternion_to_euler(x, y, z, w, degrees=False):
    """
    Convert quaternion (x, y, z, w) to Euler angles (pitch, roll, yaw) using NumPy.
    Rotation order is 'xyz'.

    Returns:
        pitch (float), roll (float), yaw (float)
    """
    # Normalize the quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Roll (x-axis rotation)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    if degrees:
        return float(np.degrees(pitch)), float(np.degrees(roll)), float(np.degrees(yaw))
    else:
        return float(pitch), float(roll), float(yaw)

class Particle:
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)

    def pose(self):
        return np.array([self.x, self.y, self.theta])

class PythonSlamNode(Node):
    def __init__(self):
        super().__init__('python_slam_node')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        # TODO: define map resolution, width, height, and number of particles
        self.declare_parameter('map_resolution', 0.1)
        self.declare_parameter('map_width_meters', 5.0)
        self.declare_parameter('map_height_meters', 5.0)

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value # Size of each map cell in meters
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0 # Map origin x-coordinate in meters
        self.map_origin_y = -5.0 # Map origin y-coordinate in meters

        # TODO: define the log-odds criteria for free and occupied cells





        self.current_pose = None
        self.log_odds_max = 4.0
        self.log_odds_min = -4.0

        self.log_odds_occupied = 0.9
        self.log_odds_free = -0.9  
        # Particle filter
        self.declare_parameter('num_particles', 10)
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        self.last_odom = None

        # ROS2 publishers/subscribers
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data)

        self.get_logger().info("Python SLAM node with particle filter initialized.")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg: Odometry):
        # Store odometry for motion update
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        if self.last_odom is None:
            return

        # 1. Motion update (sample motion model)
        odom = self.last_odom
        # TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion
        pos = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        # _, _, yaw = tf_transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        _, _, yaw = quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)
        odom_pose = np.array([pos.x, pos.y, yaw])

        # TODO: Model the particles around the current pose
        for p in self.particles:
            # Add noise to simulate motion uncertainty
            p.x = odom_pose[0] + np.random.normal(0, 0.01)
            p.y = odom_pose[1] + np.random.normal(0, 0.01)
            p.theta = odom_pose[2] + np.random.normal(0, 0.01)


        # TODO: 2. Measurement update (weight particles)
        weights = []
        for p in self.particles:
            weight = self.compute_weight(p, msg) # Compute weights for each particle
            weights.append(weight)
            # Save, append

        # Normalize weights
        N = len(weights)
        S = sum(weights)
        weights=[weights[j]/S for j in range(N)]


        for i, p in enumerate(self.particles):
            p.weight = weights[i] # Resave weights

        # 3. Resample
        self.particles = self.resample_particles(self.particles, N)

        # TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)

        mean_x = np.mean([p.x * p.weight for p in self.particles])
        mean_y = np.mean([p.y * p.weight for p in self.particles])

        mean_theta = math.atan2(
            np.mean([np.sin(p.theta) * p.weight for p in self.particles]),
            np.mean([np.cos(p.theta) * p.weight for p in self.particles])
            )

        self.current_pose = np.array([mean_x, mean_y, mean_theta])

        # 5. Mapping (update map with best particle's pose)
        for p in self.particles:
            self.update_map(p, msg)

        # 6. Broadcast map->odom transform
        self.broadcast_map_to_odom(odom_pose)

    def compute_weight(self, particle, scan_msg):
        # Simple likelihood: count how many endpoints match occupied cells
        score = 0.0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            if range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or math.isnan(range_dist): # Validate min/max sensor range constraints
                continue
            # TODO: Compute the map coordinates of the endpoint: transform the scan into the map frame

            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            # Convert endpoint to map coordinates
            x = robot_x + range_dist * np.cos(robot_theta + angle)
            y = robot_y + range_dist * np.sin(robot_theta + angle)

            # Subtract map origin and divide by resolution to obtain grid coordinates
            map_x = int((x - self.map_origin_x) / self.resolution)
            map_y = int((y - self.map_origin_y) / self.resolution)

            # TODO: Use particle.log_odds_map for scoring

            # Check bounds against map limits
            if 0 <= map_x < self.map_width_cells and 0 <= map_y < self.map_height_cells:
                # If probability is above zero, the cell is treated as occupied
                if particle.log_odds_map[map_y, map_x] > 0: # Threshold can be tuned later
                    score += 1.0


        return score + 1e-6

    def resample_particles(self, particles, N):
        # TODO: Resample particles
        new_particles = []

        step = 1.0/N
        cdf_sum=0
        p_cdf=[]

        for k in range(N):
            cdf_sum = cdf_sum+particles[k].weight
            p_cdf.append(cdf_sum)

        seed = random.uniform(0, step)

        last_index = 0
        for h in range(N):
            while seed > p_cdf[last_index]:
                last_index+=1
            new_particles.append(deepcopy(particles[last_index]))
            seed = seed+step

        return new_particles

    def update_map(self, particle, scan_msg):
        """
        Update each LiDAR scan in the global frame with map memory.
        Occupied and free evidence is accumulated in log-odds form over time.
        Values converge toward free-space or occupied-space confidence.
        """

        # Compute robot pose in map grid coordinates
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        map_x0 = int((robot_x - self.map_origin_x) / self.resolution)
        map_y0 = int((robot_y - self.map_origin_y) / self.resolution)
        for i, range_dist in enumerate(scan_msg.ranges):
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            if math.isnan(current_range) or current_range < scan_msg.range_min:
                continue

            # Convert each ray endpoint into map coordinates
            angle = robot_theta + scan_msg.angle_min + i * scan_msg.angle_increment
            x_end = robot_x + current_range * np.cos(angle)
            y_end = robot_y + current_range * np.sin(angle)
            map_x1 = int((x_end - self.map_origin_x) / self.resolution)
            map_y1 = int((y_end - self.map_origin_y) / self.resolution)
            

            # Use bresenham_line to update free cells along the ray path
            # TODO: Use self.bresenham_line for free cells
            self.bresenham_line(particle, map_x0, map_y0, map_x1, map_y1)
            # Mark endpoint cell as occupied if an obstacle was hit

            # TODO: Update particle.log_odds_map accordingly
            if is_hit:
                if 0 <= map_x1 < self.map_width_cells and 0 <= map_y1 < self.map_height_cells:
                    # Increase log-odds when a valid obstacle hit is detected
                    particle.log_odds_map[map_y1, map_x1] += self.log_odds_occupied
                    particle.log_odds_map[map_y1, map_x1] = np.clip(
                    particle.log_odds_map[map_y1, map_x1], self.log_odds_min, self.log_odds_max)


    def bresenham_line(self, particle, x0, y0, x1, y1):
        """
        Implements Bresenham's algorithm to trace a line between two grid points.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                particle.log_odds_map[y0, x0] += self.log_odds_free
                particle.log_odds_map[y0, x0] = np.clip(particle.log_odds_map[y0, x0], self.log_odds_min, self.log_odds_max)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        # Use the map from the highest-weight particle
        best_particle = max(self.particles, key=lambda p: p.weight)

        # Build occupancy grid matrix
        occ_grid = np.full(best_particle.log_odds_map.shape, -1, dtype=np.int8)
        occ_grid[best_particle.log_odds_map > 0.0] = 100
        occ_grid[best_particle.log_odds_map < 0.0] = 0
        
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()

        # Define reference frame
        map_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value

        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        map_msg.data = occ_grid.flatten().tolist()

        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Map published.")

    def broadcast_map_to_odom(self, odom_pose):
        # TODO: Broadcast map->odom transform
        t = TransformStamped()
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value

        # Estimated robot pose in map frame
        map_x, map_y, map_theta = self.current_pose

        # Robot pose from odometry
        odom_x = odom_pose[0]
        odom_y = odom_pose[1]

        # Difference between map and odometry frames (map->odom transform)
        dx = map_x - odom_x
        dy = map_y - odom_y
        dtheta = map_theta - odom_pose[2]

        # Set translation
        t.transform.translation.x = dx
        t.transform.translation.y = dy
        t.transform.translation.z = 0.0

        # Set rotation
        
        quat = quaternion_from_yaw(dtheta)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def angle_diff(a, b):
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d

def main(args=None):
    rclpy.init(args=args)
    node = PythonSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
