import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import numpy as np
from scipy.spatial.transform import Rotation as R
from enum import Enum
from geometry_msgs.msg import Pose, Point, PoseWithCovarianceStamped, TransformStamped, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from scipy import ndimage
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
import heapq

class AStarNode:
    def __init__(self, position, parent=None):
        self.parent = parent
        self.position = position # Tupla (x, y) en coordenadas de la grilla
        self.g = 0 # Costo desde el inicio hasta el nodo actual
        self.h = 0 # Heurística: costo estimado desde el nodo actual hasta el final
        self.f = 0 # Costo total (g + h)

    def __eq__(self, other):
        return self.position == other.position

    # Esto es necesario para que heapq funcione como una cola de prioridad (min-heap)
    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

def quaternion_to_euler(x, y, z, w, degrees=False):
    # Convierte quaternion a euler
    norm = np.sqrt(x*x + y*y + z*z + w*w) # Normalize the quaternion
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
        # TODO: Setear valores default
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('alpha1', 0.2)  # Error rotacional debido a rotación
        self.declare_parameter('alpha2', 0.2)  # Error rotacional debido a traslación
        self.declare_parameter('alpha3', 0.2)  # Error traslacional debido a traslación
        self.declare_parameter('alpha4', 0.2)  # Error traslacional debido a rotación
        self.declare_parameter('z_hit', 0.95)  # Peso del modelo de hit
        self.declare_parameter('z_rand', 0.05)  # Peso del modelo aleatorio
        self.declare_parameter('lookahead_distance', 0.4)
        self.declare_parameter('linear_velocity', 0.15)
        self.declare_parameter('goal_tolerance', 0.1)     # Más preciso
        self.declare_parameter('path_pruning_distance', 0.1)  # Distancia para podar path
        self.declare_parameter('safety_margin_cells', 2)  # Margen de seguridad en celdas
        
        
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
        self.grid = None
        # Agregar
        self.current_path_index = 0

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
        # TODO: Inicializar particulas en base a la pose inicial con variaciones aleatorias
        # Deben ser la misma cantidad de particulas que self.num_particles
        # Deben tener un peso
        for idx in range(self.num_particles):
            orientation = initial_pose.orientation
            poseX = initial_pose.position.x + np.random.normal(0,0.02)
            poseY = initial_pose.position.y + np.random.normal(0,0.02)
            _, _, theta = quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)
            self.particles[idx] = np.array([poseX, poseY, theta + np.random.normal(0,0.02)])
        
        # self.weights = np.ones(self.num_particles) / self.num_particles
        self.publish_particles()

    def initialize_particles_randomly(self):
        # TODO: Inizializar particulas aleatoriamente en todo el mapa
        
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        map_width_meters = self.map_data.info.width * self.map_data.info.resolution
        map_height_meters = self.map_data.info.height * self.map_data.info.resolution

        for idx in range(self.num_particles):
            poseX = np.random.uniform(origin_x, origin_x + map_width_meters)
            poseY = np.random.uniform(origin_y, origin_y + map_height_meters)
            rand_theta = np.random.uniform(-np.pi, np.pi)
            self.particles[idx] = np.array([poseX, poseY, rand_theta])


        self.publish_particles()

    def stop_robot(self):
        """Publica un mensaje Twist con todas las velocidades a cero."""
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)

    def check_for_imminent_obstacle(self):
        """
        Verifica el escaneo láser en un cono frontal para detectar obstáculos cercanos.
        """
        if self.latest_scan is None:
            return False

        # Defino el cono de detección frontal (ej. +/- 30 grados)
        front_angle_range = np.deg2rad(30)
        
        for i, dist in enumerate(self.latest_scan.ranges):
            angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
            if abs(angle) < front_angle_range / 2.0:
                if 0 < dist < self.obstacle_detection_distance:
                    return True
        return False

    def create_path_message(self, grid_path):
        """Convierte path en coordenadas de grilla a mensaje Path de ROS."""
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame_id
        
        for grid_point in grid_path:
            # Convertir coordenadas de grilla a mundo
            world_x, world_y = self.grid_to_world(grid_point[0], grid_point[1])
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.map_frame_id
            pose_stamped.pose.position.x = world_x
            pose_stamped.pose.position.y = world_y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0  # Sin rotación específica
            
            path_msg.poses.append(pose_stamped)
        
        return path_msg

    def smooth_path(self, path, iterations=3):
        """Suavizar path usando promedio móvil"""
        if len(path) <= 2:
            return path
            
        smoothed = path.copy()
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                # Promedio de punto anterior, actual y siguiente
                prev_point = np.array(smoothed[i-1])
                curr_point = np.array(smoothed[i])
                next_point = np.array(smoothed[i+1])
                
                new_point = (prev_point + curr_point + next_point) / 3.0
                
                # Verificar que el punto suavizado no esté en obstáculo
                gx, gy = self.world_to_grid(new_point[0], new_point[1])
                if (0 <= gx < self.map_data.info.width and 
                    0 <= gy < self.map_data.info.height and
                    self.inflated_grid[gy, gx] < 50):
                    smoothed[i] = new_point.tolist()
        
        return smoothed

    def timer_callback(self):
        # TODO: Implementar maquina de estados para cada caso.
        # Debe haber estado para PLANNING, NAVIGATING y AVOIDING_OBSTACLE, pero pueden haber más estados si se desea.
        if not self.map_received:
            return

        # --- Localization (always running) ---
        if self.latest_scan is None:
            return
            
        if not self.initial_pose_received:
            self.initialize_particles_randomly()
            self.initial_pose_received = True
            return

        # 1- Obtengo la posición del Robot
        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            # Acá se detiene el robot si se pierde la odometría
            if self.state in [State.NAVIGATING, State.AVOIDING_OBSTACLE]:
                self.stop_robot()
            return

        # Actualizo la pose de las particulas
        if self.last_odom_pose is not None:
            self.motion_model(current_odom_tf)


        self.measurement_model()

        # Resampleo si es necesario
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        if effective_sample_size < self.num_particles / 2:
            self.resample()

        estimated_pose = self.estimate_pose()

        # CASO 1: Si está localizado pero sin objetivo lo detengo
        if self.state == State.IDLE:
            self.stop_robot()

        elif self.state == State.PLANNING:
            path = self.A_algorithm(estimated_pose, self.goal_pose)

            if path:
                smoothed_path = self.smooth_path(path)

                self.get_logger().info(f'Ruta encontrada: {path}. State -> NAVIGATING')
                self.current_path = smoothed_path
                self.current_path_index = 0 
                self.publish_path(self.create_path_message(smoothed_path))
                self.state = State.NAVIGATING
            else:
                self.get_logger().error('No se pudo encontrar una ruta al objetivo. State -> IDLE')
                self.state = State.IDLE
        
        # CASO 2: Tiene una ruta y la esta siguiendo
        elif self.state == State.NAVIGATING:
            # Acá verifico si hay un obstaculo inminente
            if self.check_for_imminent_obstacle():
                self.get_logger().warn('Obstáculo detectado! State -> AVOIDING_OBSTACLE')
                self.stop_robot() # Detenerse antes de girar
                self.state = State.AVOIDING_OBSTACLE
                return
            
            self.follow_path(estimated_pose)

        # CASO 3: Tiene un obstaculo inminente

        elif self.state == State.AVOIDING_OBSTACLE:
            if not self.check_for_imminent_obstacle():
                self.get_logger().info('El camino está despejado. Reanudando navegación. State -> NAVIGATING')
                self.state = State.NAVIGATING
            
            else:
                # Maniobra simple: girar en el lugar.
                twist_msg = Twist()
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = self.obstacle_avoidance_turn_speed # Girar en sentido antihorario
                self.cmd_vel_pub.publish(twist_msg)


        # TODO: Implementar codigo para publicar la pose estimada, las particulas, y la transformacion entre el mapa y la base del robot.

        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)

    def follow_path(self, estimated_pose):
        robot_pos = np.array([estimated_pose.position.x, estimated_pose.position.y])
        _, _, robot_yaw = quaternion_to_euler(
            estimated_pose.orientation.x, 
            estimated_pose.orientation.y, 
            estimated_pose.orientation.z, 
            estimated_pose.orientation.w
        )

        if self.current_path is None or len(self.current_path) == 0:
            self.stop_robot()
            self.state = State.IDLE
            return

        goal_pos = np.array(self.current_path[-1])
        if np.linalg.norm(robot_pos - goal_pos) < self.goal_tolerance:
            self.get_logger().info("¡Objetivo alcanzado!")
            self.state = State.IDLE
            self.stop_robot()
            return
        
        # SOLUCIÓN 1: Buscar punto más cercano en el path
        min_dist = float('inf')
        closest_index = self.current_path_index
        
        for i in range(max(0, self.current_path_index - 2), len(self.current_path)):
            path_point = np.array(self.current_path[i])
            dist = np.linalg.norm(robot_pos - path_point)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        self.current_path_index = max(self.current_path_index, closest_index)
        
        # SOLUCIÓN 2: Usar punto lookahead más inteligente
        lookahead_point = None
        target_distance = max(self.lookahead_distance, 0.2)  # Mínimo 20cm
        
        for i in range(self.current_path_index, len(self.current_path)):
            path_point = np.array(self.current_path[i])
            dist_to_point = np.linalg.norm(robot_pos - path_point)
            
            if dist_to_point >= target_distance:
                lookahead_point = path_point
                break
        
        # Si no hay punto lookahead, usar el siguiente punto en el path
        if lookahead_point is None:
            if self.current_path_index + 1 < len(self.current_path):
                lookahead_point = np.array(self.current_path[self.current_path_index + 1])
            else:
                lookahead_point = goal_pos
        
        # SOLUCIÓN 3: Control mejorado con zona muerta angular
        angle_to_target = np.arctan2(
            lookahead_point[1] - robot_pos[1], 
            lookahead_point[0] - robot_pos[0]
        )
        angle_error = angle_to_target - robot_yaw
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # ZONA MUERTA ANGULAR - Evita oscilaciones
        angle_threshold = np.deg2rad(10)  # 10 grados de zona muerta
        
        if abs(angle_error) < angle_threshold:
            # Si el ángulo es pequeño, ir hacia adelante con corrección mínima
            angular_vel = 0.5 * angle_error  # Ganancia muy baja
            linear_vel = self.linear_velocity
        elif abs(angle_error) > np.deg2rad(90):  # Si necesita girar mucho
            # Rotar en el lugar sin moverse hacia adelante
            angular_vel = 0.8 * np.sign(angle_error)  # Velocidad angular constante
            linear_vel = 0.0
        else:
            # Control proporcional normal
            angular_vel = 1.0 * angle_error
            linear_vel = self.linear_velocity * (1.0 - abs(angle_error) / np.pi)
        
        # Limitar velocidades
        angular_vel = float(np.clip(angular_vel, -1.0, 1.0))
        linear_vel = float(np.clip(linear_vel, 0.0, self.linear_velocity))
        
        # SOLUCIÓN 4: Evitar comandos de velocidad muy pequeños
        if abs(angular_vel) < 0.05:
            angular_vel = 0.0
        if linear_vel < 0.02:
            linear_vel = 0.0
        
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)


    # def follow_path(self, estimated_pose):
    #     robot_pos = np.array([estimated_pose.position.x, estimated_pose.position.y])
    #     _, _, robot_yaw = quaternion_to_euler(
    #         estimated_pose.orientation.x, 
    #         estimated_pose.orientation.y, 
    #         estimated_pose.orientation.z, 
    #         estimated_pose.orientation.w
    #     )

    #     # Verificar si llegamos al objetivo
    #     if self.current_path is None or len(self.current_path) == 0:
    #         self.stop_robot()
    #         self.state = State.IDLE
    #         return

    #     goal_pos = np.array(self.current_path[-1])
    #     if np.linalg.norm(robot_pos - goal_pos) < self.goal_tolerance:
    #         self.get_logger().info("¡Objetivo alcanzado!")
    #         self.state = State.IDLE
    #         self.stop_robot()
    #         return
        
    #     # SOLUCIÓN: Mejorar lógica de seguimiento de path
    #     # Encontrar el punto más cercano en el path
    #     min_dist = float('inf')
    #     closest_index = self.current_path_index
        
    #     for i in range(self.current_path_index, len(self.current_path)):
    #         path_point = np.array(self.current_path[i])
    #         dist = np.linalg.norm(robot_pos - path_point)
    #         if dist < min_dist:
    #             min_dist = dist
    #             closest_index = i
        
    #     # Actualizar índice solo si avanzamos
    #     if closest_index > self.current_path_index:
    #         self.current_path_index = closest_index
        
    #     # Encontrar punto lookahead
    #     lookahead_point = None
    #     for i in range(self.current_path_index, len(self.current_path)):
    #         path_point = np.array(self.current_path[i])
    #         if np.linalg.norm(robot_pos - path_point) >= self.lookahead_distance:
    #             lookahead_point = path_point
    #             break
        
    #     # Si no hay punto lookahead, usar el objetivo final
    #     if lookahead_point is None:
    #         lookahead_point = goal_pos
        
    #     # Calcular control
    #     angle_to_lookahead = np.arctan2(
    #         lookahead_point[1] - robot_pos[1], 
    #         lookahead_point[0] - robot_pos[0]
    #     )
    #     angle_error = angle_to_lookahead - robot_yaw
        
    #     # Normalizar ángulo entre -π y π
    #     angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

    #     # SOLUCIÓN: Control proporcional mejorado
    #     # Si el error angular es grande, reducir velocidad lineal
    #     angular_vel = 2.0 * angle_error
    #     linear_vel = self.linear_velocity
        
    #     if abs(angle_error) > np.pi/4:  # Si error > 45 grados
    #         linear_vel *= 0.3  # Reducir velocidad lineal
    #     elif abs(angle_error) > np.pi/6:  # Si error > 30 grados
    #         linear_vel *= 0.6

    #     # Publicar comando
    #     twist_msg = Twist()
    #     twist_msg.linear.x = linear_vel
    #     twist_msg.angular.z = float(np.clip(angular_vel, -1.5, 1.5))
    #     self.cmd_vel_pub.publish(twist_msg)


    # def follow_path(self, estimated_pose):
    #     robot_pos = np.array([estimated_pose.position.x, estimated_pose.position.y])
    #     _, _, robot_yaw = quaternion_to_euler(estimated_pose.orientation.x, estimated_pose.orientation.y, estimated_pose.orientation.z, estimated_pose.orientation.w)

    #     # Verificar si hemos llegado al final de la ruta
    #     goal_pos = np.array(self.current_path[-1])
    #     if np.linalg.norm(robot_pos - goal_pos) < self.goal_tolerance:
    #         self.get_logger().info("¡Objetivo alcanzado!")
    #         self.state = State.IDLE
    #         self.stop_robot()
    #         return
        
    #     # Encontrar el punto "lookahead" en la ruta
    #     lookahead_point = None
    #     for i in range(self.current_path_index, len(self.current_path)):
    #         path_point = np.array(self.current_path[i])
    #         if np.linalg.norm(robot_pos - path_point) > self.lookahead_distance:
    #             lookahead_point = path_point
    #             self.current_path_index = i # Actualizar índice para no re-evaluar partes pasadas de la ruta
    #             break
        
    #     # Si no se encuentra un punto lejano (estamos cerca del final), tomar el último punto.
    #     if lookahead_point is None:
    #         lookahead_point = goal_pos
        
        
    #     # Calcular el error de ángulo hacia el punto lookahead
    #     angle_to_lookahead = np.arctan2(lookahead_point[1] - robot_pos[1], lookahead_point[0] - robot_pos[0])
    #     angle_error = angle_to_lookahead - robot_yaw
        
    #     # Normalizar ángulo
    #     angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

    #     # Calcular velocidad angular (control proporcional simple)
    #     angular_vel = 2.0 * angle_error # Ganancia P = 2.0

    #     # Publicó el comando de velocidad
    #     twist_msg = Twist()
    #     twist_msg.linear.x = self.linear_velocity
    #     twist_msg.angular.z = float(np.clip(angular_vel, -1.0, 1.0)) # Limitar velocidad angular
    #     self.cmd_vel_pub.publish(twist_msg)

    def get_heuristic(self, cell, goal):
        
        y_c, x_c = cell
        y_g, x_g = goal

        dy = y_c - y_g
        dx = x_c - x_g

        heuristic = np.hypot(dy, dx)

        return heuristic

    def get_neighborhood(self, cell, occ_map_shape):

        fila, col = cell
        filas_totales, columnas_totales = occ_map_shape

        movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        vecinos = []
        for df, dc in movimientos:
            nueva_fila, nueva_col = fila + df, col + dc
            if 0 <= nueva_fila < filas_totales and 0 <= nueva_col < columnas_totales:
                vecinos.append((nueva_fila, nueva_col))

        return vecinos
    
    def get_edge_cost(self, parent, child):
        fila_p, col_p = parent
        fila_c, col_c = child
        
        # SOLUCIÓN: Usar mapa inflado en lugar del original
        if self.inflated_grid[fila_c, col_c] >= 50:  # Obstáculo en mapa inflado
            return float('inf')

        # Costo euclidiano
        if abs(fila_p - fila_c) + abs(col_p - col_c) == 2:  # Movimiento diagonal
            return 1.414  # sqrt(2)
        else:
            return 1.0    # Movimiento cardinal


    # def get_edge_cost(self, parent, child):
    #     y_p, x_p = parent
    #     y_c, x_c = child
    #     p_ocup = self.grid[y_c, x_c]

    #     if p_ocup > 0.4:
    #         return float('inf')

    #     p_p = np.array([x_p, y_p])
    #     p_c = np.array([x_c, y_c])

    #     distancia = np.linalg.norm(p_p - p_c)

    #     edge_cost = distancia * (1.0 + float(p_ocup))

    #     return edge_cost

    # Bien
    # def A_algorithm(self, initial_pose, goal_point):
    #     start_grid = self.world_to_grid(initial_pose[0], initial_pose[1])
    #     goal_grid = self.world_to_grid(goal_point[0], goal_point[1])
        
    #     # PROBLEMA: start_grid y goal_grid devuelven (gx, gy) pero luego usas [1],[0]
    #     # SOLUCIÓN: Ser consistente
    #     start_cell = (start_grid[1], start_grid[0])  # (fila, columna)
    #     goal_cell = (goal_grid[1], goal_grid[0])    # (fila, columna)
        
    #     # Verificar si inicio o final están en obstáculo
    #     if self.inflated_grid[start_cell[0], start_cell[1]] == 100 or \
    #     self.inflated_grid[goal_cell[0], goal_cell[1]] == 100:
    #         self.get_logger().warn("El punto de inicio o final está dentro de un obstáculo.")
    #         return None
        
    #     costs = np.ones(self.grid.shape) * np.inf
    #     closed_flags = np.zeros(self.grid.shape)
    #     predecessors = -np.ones(self.grid.shape + (2,), dtype=np.int64)

    #     # Calcular heurística para toda la grilla
    #     heuristic = np.zeros(self.grid.shape)
    #     for fila in range(self.grid.shape[0]):
    #         for col in range(self.grid.shape[1]):
    #             heuristic[fila, col] = self.get_heuristic((fila, col), goal_cell)

    #     parent = start_cell
    #     costs[start_cell[0], start_cell[1]] = 0

    #     while not np.array_equal(parent, goal_cell):
    #         open_costs = np.where(closed_flags == 1, np.inf, costs) + heuristic
    #         fila, col = np.unravel_index(open_costs.argmin(), open_costs.shape)

    #         if open_costs[fila, col] == np.inf:
    #             break

    #         parent = (fila, col)
    #         closed_flags[fila, col] = 1

    #         neighbors = self.get_neighborhood(parent, self.grid.shape)
    #         for neighbor in neighbors:
    #             n_fila, n_col = neighbor

    #             edge_cost = self.get_edge_cost(parent, neighbor)
    #             if edge_cost == float('inf'):
    #                 continue

    #             new_cost = costs[parent[0], parent[1]] + edge_cost

    #             if new_cost < costs[n_fila, n_col]:
    #                 costs[n_fila, n_col] = new_cost
    #                 predecessors[n_fila, n_col] = parent

    #     # Reconstruir path
    #     if np.array_equal(parent, goal_cell):
    #         path = []
    #         current = goal_cell
    #         while predecessors[current[0], current[1]][0] >= 0:
    #             # Convertir de (fila, col) a (gx, gy) para world_to_grid
    #             gx, gy = current[1], current[0]
    #             world_x, world_y = self.grid_to_world(gx, gy)
    #             path.append([world_x, world_y])
    #             current = tuple(predecessors[current[0], current[1]])
            
    #         # Agregar punto inicial
    #         gx, gy = start_cell[1], start_cell[0]
    #         world_x, world_y = self.grid_to_world(gx, gy)
    #         path.append([world_x, world_y])
            
    #         path.reverse()
    #         return path
    #     else:
    #         return None

    def A_algorithm(self, start_pose, goal_pose):
        """
        Planifica una ruta desde una pose de inicio a una pose objetivo usando el algoritmo A*.
        :param start_pose: Pose de inicio (geometry_msgs/Pose).
        :param goal_pose: Pose objetivo (geometry_msgs/Pose).
        :return: Una lista de tuplas (x, y) con las coordenadas del mundo si se encuentra ruta, de lo contrario None.
        """
        self.get_logger().info("A* planning started.")
        
        start_grid = self.world_to_grid(start_pose.position.x, start_pose.position.y)
        goal_grid = self.world_to_grid(goal_pose.position.x, goal_pose.position.y)

        map_h, map_w = self.inflated_grid.shape
        # if not (0 <= start_grid[0] < map_w and 0 <= start_grid[1] < map_h) or not (0 <= goal_grid[0] < map_w and 0 <= goal_grid[1] < map_h) or self.inflated_grid[start_grid[1], start_grid[0]] == 100 or self.inflated_grid[goal_grid[1], goal_grid[0]] == 100:
        #     self.get_logger().error("A* failed: Invalid start or goal position.")
        #     return None

        start_node = AStarNode(start_grid)
        goal_node = AStarNode(goal_grid)
        
        open_list, closed_list = [], set()
        heapq.heappush(open_list, start_node)

        while open_list:
            current_node = heapq.heappop(open_list)
            if current_node in closed_list: continue
            closed_list.add(current_node)

            if current_node == goal_node:
                # --- Reconstrucción de la ruta ---
                grid_path = []
                current = current_node
                while current is not None:
                    grid_path.append(current.position)
                    current = current.parent
                grid_path = grid_path[::-1] # Invertir para que vaya del inicio al fin
                
                # --- MODIFICACIÓN: Convertir a lista de coordenadas del mundo ---
                world_path = []
                for point in grid_path:
                    world_coords = self.grid_to_world(point[0], point[1])
                    world_path.append(world_coords)

                self.get_logger().info("A* path found.")
                return world_path # Devolver la lista de tuplas (x, y)

            # Expansión de vecinos... (sin cambios)
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                if not (0 <= node_position[0] < map_w and 0 <= node_position[1] < map_h) or \
                   self.inflated_grid[node_position[1], node_position[0]] == 100:
                    continue

                neighbor = AStarNode(node_position, parent=current_node)
                if neighbor in closed_list: continue

                move_cost = 1.414 if new_position[0] != 0 and new_position[1] != 0 else 1.0
                neighbor.g = current_node.g + move_cost
                neighbor.h = ((neighbor.position[0] - goal_node.position[0]) ** 2) + ((neighbor.position[1] - goal_node.position[1]) ** 2)
                neighbor.f = neighbor.g + neighbor.h
                
                heapq.heappush(open_list, neighbor)

        self.get_logger().warn("A* failed to find a path. The open list is empty.")
        return None

    # def A_algorithm(self, initial_pose, goal_point):
    #     start_grid = self.world_to_grid(initial_pose[0], initial_pose[1])
    #     goal_grid = self.world_to_grid(goal_point[0], goal_point[1])
        
    #     # Por si esta el inicio o el final en un obstaculo
    #     if self.inflated_grid[start_grid[1], start_grid[0]] == 100 or \
    #         self.inflated_grid[goal_grid[1], goal_grid[0]] == 100:
    #         self.get_logger().warn("El punto de inicio o final está dentro de un obstáculo.")
    #         return None
        
    #     costs = np.ones(self.grid.shape) * np.inf
        
    #     closed_flags = np.zeros(self.grid.shape)

    #     predecessors = -np.ones(self.grid.shape + (2,), dtype=np.int64)

    #     heuristic = np.zeros(self.grid.shape)
    #     for x in range(self.grid.shape[0]):
    #         for y in range(self.grid.shape[1]):
    #             heuristic[x, y] = self.get_heuristic([x, y], goal_grid) * 2 # Factor que puedo cambiar 

    #     parent = start_grid
    #     costs[start_grid[0], start_grid[1]] = 0

    #     while not np.array_equal(parent, goal_grid):

    #         open_costs = np.where(closed_flags==1, np.inf, costs) + heuristic
            
    #         x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)

    #         if open_costs[x, y] == np.inf:
    #             break

    #         parent = np.array([x, y])
    #         closed_flags[x, y] = 1

    #         neighbors = self.get_neighborhood(parent, self.grid.shape)
    #         for neighbor in neighbors:
    #             y_n, x_n = neighbor

    #             edge_cost = self.get_edge_cost(parent, neighbor)
    #             if edge_cost == float('inf'):
    #                 continue

    #             new_cost = costs[parent[0], parent[1]] + edge_cost

    #             if new_cost < costs[y_n, x_n]:
    #                 costs[y_n, x_n] = new_cost
    #                 predecessors[y_n, x_n] = parent 


    #     if np.array_equal(parent, goal_grid):
    #         path = []
    #         while predecessors[parent[0], parent[1]][0] >= 0:
    #             path.append(parent.tolist())
    #             predecessor = predecessors[parent[0], parent[1]]
    #             parent = predecessor      

    #         path.append(start_grid)
    #         path.reverse()   

    #         world_path = []
    #         for grid_point in path:
    #             world_x, world_y = self.grid_to_world(grid_point[0], grid_point[1])
    #             world_path.append([world_x, world_y])

    #         return world_path
    
    #     else: return None

    def get_odom_transform(self):
        try:
            return self.tf_buffer.lookup_transform(self.odom_frame_id, self.base_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'Could not get transform from {self.odom_frame_id} to {self.base_frame_id}. Skipping update. Error: {e}', throttle_duration_sec=2.0)
            return None

    # Bien
    def motion_model(self, current_odom_tf):
        # TODO: Implementar el modelo de movimiento para actualizar las particulas.

        current_odom_pose = current_odom_tf.transform
        
        # Si es la primera vez, solo guardamos la pose actual
        if self.last_odom_pose is None:
            self.last_odom_pose = current_odom_pose
            return
        
        # Extraer posiciones y orientaciones anteriores
        x_prev = self.last_odom_pose.translation.x
        y_prev = self.last_odom_pose.translation.y
        _, _, theta_prev = quaternion_to_euler(
            self.last_odom_pose.rotation.x,
            self.last_odom_pose.rotation.y, 
            self.last_odom_pose.rotation.z,
            self.last_odom_pose.rotation.w
        )
        
        # Extraer posiciones y orientaciones actuales
        x_curr = current_odom_pose.translation.x
        y_curr = current_odom_pose.translation.y
        _, _, theta_curr = quaternion_to_euler(
            current_odom_pose.rotation.x,
            current_odom_pose.rotation.y,
            current_odom_pose.rotation.z, 
            current_odom_pose.rotation.w
        )
        
        # Calcular el movimiento en odometría
        delta_x = x_curr - x_prev
        delta_y = y_curr - y_prev
        delta_theta = theta_curr - theta_prev
        
        # Normalizar el ángulo delta_theta
        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
        
        # Si no hay movimiento significativo, no actualizar partículas
        delta_trans = np.sqrt(delta_x**2 + delta_y**2)
        if delta_trans < 1e-6 and abs(delta_theta) < 1e-6:
            self.last_odom_pose = current_odom_pose
            return
        
        # Calcular los componentes del movimiento (modelo de odometría)
        delta_rot1 = np.arctan2(delta_y, delta_x) - theta_prev
        delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))  # Normalizar
        
        delta_rot2 = delta_theta - delta_rot1
        delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))  # Normalizar
        
        # Actualizar cada partícula con ruido
        for i in range(self.num_particles):
            # Agregar ruido basado en los parámetros alpha
            # El ruido es proporcional a la magnitud del movimiento
            
            # Ruido en la primera rotación
            delta_rot1_noisy = delta_rot1 + np.random.normal(
                0, self.alphas[0] * abs(delta_rot1) + self.alphas[1] * delta_trans
            )
            
            # Ruido en la traslación
            delta_trans_noisy = delta_trans + np.random.normal(
                0, self.alphas[2] * delta_trans + self.alphas[3] * (abs(delta_rot1) + abs(delta_rot2))
            )
            
            # Ruido en la segunda rotación
            delta_rot2_noisy = delta_rot2 + np.random.normal(
                0, self.alphas[0] * abs(delta_rot2) + self.alphas[1] * delta_trans
            )
            
            # Aplicar el movimiento a la partícula
            x_particle = self.particles[i, 0]
            y_particle = self.particles[i, 1] 
            theta_particle = self.particles[i, 2]
            
            # Actualizar la partícula
            # Primer: rotar hacia la dirección del movimiento
            theta_particle += delta_rot1_noisy
            
            # Segundo: trasladarse en esa dirección
            x_particle += delta_trans_noisy * np.cos(theta_particle)
            y_particle += delta_trans_noisy * np.sin(theta_particle)
            
            # Tercero: rotar para ajustar la orientación final
            theta_particle += delta_rot2_noisy
            
            # Normalizar el ángulo
            theta_particle = np.arctan2(np.sin(theta_particle), np.cos(theta_particle))
            
            # Guardar la partícula actualizada
            self.particles[i, 0] = x_particle
            self.particles[i, 1] = y_particle
            self.particles[i, 2] = theta_particle
        
        # Actualizar la pose anterior
        self.last_odom_pose = current_odom_pose

    # Bien
    def measurement_model(self):
        map_res = self.map_data.info.resolution
        map_origin = self.map_data.info.origin.position
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        map_img = np.array(self.map_data.data).reshape((map_h, map_w))

        # Parámetros del láser
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = np.array(self.latest_scan.ranges)
        
        # Usar solo cada N-ésima medida para eficiencia computacional
        skip = max(1, len(ranges) // 50)  # Usar aproximadamente 50 rayos
        
        # Parámetros del modelo de medición
        sigma_hit = 0.2  # Desviación estándar del modelo de hit
        
        # Procesar cada partícula
        for i in range(self.num_particles):
            px, py, ptheta = self.particles[i]
            
            # Verificar si la partícula está dentro del mapa
            gx = int((px - map_origin.x) / map_res)
            gy = int((py - map_origin.y) / map_res)
            
            if gx < 0 or gx >= map_w or gy < 0 or gy >= map_h:
                self.weights[i] = 1e-10  # Peso muy bajo para partículas fuera del mapa
                continue
            
            # Verificar si la partícula está en un obstáculo
            if map_img[gy, gx] > 50:  # Obstáculo
                self.weights[i] = 1e-10
                continue
                
            weight = 1.0
            valid_measurements = 0
            
            # Procesar mediciones del láser con submuestreo
            for j in range(0, len(ranges), skip):
                z_measured = ranges[j]
                
                # Filtrar mediciones inválidas
                if not np.isfinite(z_measured) or z_measured <= 0 or z_measured > self.laser_max_range:
                    continue
                    
                # Ángulo del rayo en el frame del robot
                ray_angle = angle_min + j * angle_increment
                # Ángulo global del rayo
                global_angle = ptheta + ray_angle
                
                # Calcular distancia esperada usando ray casting
                z_expected = self.ray_cast(px, py, global_angle, map_img, map_res, map_origin, map_w, map_h)
                
                if z_expected > 0:
                    # Modelo de likelihood con componentes hit y random
                    diff = abs(z_measured - z_expected)
                    
                    # Componente hit (gaussiana centrada en la medición esperada)
                    p_hit = (1.0 / (sigma_hit * np.sqrt(2 * np.pi))) * \
                            np.exp(-0.5 * (diff / sigma_hit)**2)
                    
                    # Componente random (uniforme)
                    p_rand = 1.0 / self.laser_max_range
                    
                    # Probabilidad total
                    prob = self.z_hit * p_hit + self.z_rand * p_rand
                    
                    # Multiplicar probabilidades (asumiendo independencia)
                    weight *= prob
                    valid_measurements += 1
            
            # Asignar peso a la partícula
            if valid_measurements > 0:
                self.weights[i] = weight
            else:
                self.weights[i] = 1e-10  # Peso muy bajo si no hay mediciones válidas
        
        # Normalizar pesos
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # Si todos los pesos son cero, reinicializar uniformemente
            self.weights.fill(1.0 / self.num_particles)
            self.get_logger().warn("Todos los pesos eran cero. Reinicializando uniformemente.")

    # Bien
    def ray_cast(self, x, y, angle, map_img, map_res, map_origin, map_w, map_h):
        """
        Implementa ray casting para calcular distancia esperada del láser.
        """
        step_size = map_res / 2.0  # Paso pequeño para precisión
        max_range = self.laser_max_range
        
        dx = np.cos(angle) * step_size
        dy = np.sin(angle) * step_size
        
        current_x = x
        current_y = y
        distance = 0.0
        
        while distance < max_range:
            # Convertir a coordenadas de grilla
            gx = int((current_x - map_origin.x) / map_res)
            gy = int((current_y - map_origin.y) / map_res)
            
            # Verificar límites del mapa
            if gx < 0 or gx >= map_w or gy < 0 or gy >= map_h:
                break
                
            # Verificar si hay obstáculo
            if map_img[gy, gx] > 50:  # Obstáculo detectado
                return distance
                
            # Avanzar en la dirección del rayo
            current_x += dx
            current_y += dy
            distance += step_size
        
        return max_range  # No se encontró obstáculo

    # Bien
    def resample(self):
        # TODO: Implementar el resampleo de las particulas basado en los pesos.
        # Crear CDF de los pesos
        cumulative_sum = np.cumsum(self.weights)
        
        # Generar números aleatorios sistemáticos
        r = np.random.random() / self.num_particles
        indices = []
        
        i = 0
        for j in range(self.num_particles):
            u = r + j / self.num_particles
            while u > cumulative_sum[i]:
                i += 1
            indices.append(i)
        
        # Crear nuevas partículas basadas en los índices seleccionados
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    # Bien
    def estimate_pose(self):
        weight_sum = np.sum(self.weights)
        normalized_weights = self.weights / weight_sum

        weighted_x = np.sum(self.particles[:, 0] * normalized_weights)
        weighted_y = np.sum(self.particles[:, 1] * normalized_weights)
        
        # Acá puedo usar el promedio ponderado tambien
        cos_angles = np.cos(self.particles[:, 2])
        sin_angles = np.sin(self.particles[:, 2])

        weighted_cos = np.sum(cos_angles * normalized_weights)
        weighted_sin = np.sum(sin_angles * normalized_weights)
    
        weighted_theta = np.arctan2(weighted_sin, weighted_cos)


        estimated_pose = Pose()
    
        # Posición
        estimated_pose.position = Point()
        estimated_pose.position.x = float(weighted_x)
        estimated_pose.position.y = float(weighted_y)
        estimated_pose.position.z = 0.0  # Asumimos robot en 2D

        rotation = R.from_euler('z', weighted_theta)
        quat = rotation.as_quat()
        
        estimated_pose.orientation = Quaternion()
        estimated_pose.orientation.x = float(quat[0])
        estimated_pose.orientation.y = float(quat[1])
        estimated_pose.orientation.z = float(quat[2])
        estimated_pose.orientation.w = float(quat[3])
        
        return estimated_pose

    def publish_pose(self, estimated_pose):
        """
        Publica la pose estimada en el tópico 'amcl_pose' como PoseWithCovarianceStamped.
        """
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.map_frame_id
        pose_msg.pose.pose = estimated_pose

        # Calcular covarianza simple a partir de las partículas
        cov = np.cov(self.particles.T, aweights=self.weights)
        # Asignar solo las componentes relevantes (x, y, theta)
        pose_msg.pose.covariance[0] = cov[0, 0]  # var(x)
        pose_msg.pose.covariance[7] = cov[1, 1]  # var(y)
        pose_msg.pose.covariance[35] = cov[2, 2] # var(theta)

        self.pose_pub.publish(pose_msg)

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
        map_to_base_mat = self.pose_to_matrix(estimated_pose)
        odom_to_base_mat = self.transform_to_matrix(odom_tf.transform)
        map_to_odom_mat = np.dot(map_to_base_mat, np.linalg.inv(odom_to_base_mat))
        
        t = TransformStamped()
        
        # TODO: Completar el TransformStamped con la transformacion entre el mapa y la base del robot.

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame_id
        t.child_frame_id = self.odom_frame_id

        # Extraer traslación y rotación de la matriz
        translation = map_to_odom_mat[:3, 3]
        rotation_matrix = map_to_odom_mat[:3, :3]
        quat = R.from_matrix(rotation_matrix).as_quat()

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

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

    def inflate_map(self):
        """Inflar obstáculos para margen de seguridad"""
        if self.grid is None:
            return
            
        # Crear mapa binario de obstáculos
        obstacle_map = (self.grid > 50).astype(np.uint8)
        
        # Crear kernel de inflado
        kernel_size = 2 * self.safety_margin_cells + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Inflar obstáculos
        inflated = ndimage.binary_dilation(obstacle_map, structure=kernel)
        
        # Crear mapa inflado manteniendo valores originales
        self.inflated_grid = self.grid.copy()
        self.inflated_grid[inflated] = 100
        
        self.get_logger().info(f"Mapa inflado con margen de {self.safety_margin_cells} celdas")


    # def inflate_map(self):
    #     # Inflar obstáculos en el mapa para margen de seguridad
    #     structure = np.ones((2 * self.safety_margin_cells + 1, 2 * self.safety_margin_cells + 1))
    #     occupied = self.grid > 50
    #     inflated = ndimage.binary_dilation(occupied, structure=structure)
    #     self.grid[inflated] = 100
    #     self.inflated_grid = self.grid.copy()

def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 


    # def follow_path(self, estimated_pose):
    #     """
    #     Controlador de seguimiento de ruta con una lógica de velocidad más equilibrada
    #     para evitar quedarse girando en el lugar.
    #     """
    #     robot_pos = np.array([estimated_pose.position.x, estimated_pose.position.y])
    #     _, _, robot_yaw = quaternion_to_euler(
    #         estimated_pose.orientation.x, 
    #         estimated_pose.orientation.y, 
    #         estimated_pose.orientation.z, 
    #         estimated_pose.orientation.w
    #     )

    #     if self.current_path is None or self.current_path_index >= len(self.current_path):
    #         self.get_logger().info("Follow_path: No hay ruta o se ha completado.")
    #         self.stop_robot()
    #         self.state = State.IDLE
    #         return

    #     # Comprobar si hemos llegado al objetivo final
    #     goal_pos = np.array(self.current_path[-1])
    #     if np.linalg.norm(robot_pos - goal_pos) < self.goal_tolerance:
    #         self.get_logger().info("¡Objetivo alcanzado!")
    #         self.state = State.IDLE
    #         self.stop_robot()
    #         return
        
    #     # --- Búsqueda del punto de mira (lookahead_point) ---
    #     lookahead_point = None
    #     for i in range(self.current_path_index, len(self.current_path)):
    #         path_point = np.array(self.current_path[i])
    #         dist = np.linalg.norm(robot_pos - path_point)
    #         if dist > self.lookahead_distance:
    #             lookahead_point = path_point
    #             self.current_path_index = i
    #             break
        
    #     if lookahead_point is None:
    #         lookahead_point = goal_pos

    #     # --- Lógica de control MEJORADA ---
    #     angle_to_target = np.arctan2(lookahead_point[1] - robot_pos[1], lookahead_point[0] - robot_pos[0])
    #     angle_error = angle_to_target - robot_yaw
    #     angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
    #     # ** NUEVA LÓGICA DE VELOCIDAD **

    #     # CASO 1: El giro es muy pronunciado (ej > 100 grados). 
    #     # En este caso, sí queremos que rote en el lugar primero.
    #     if abs(angle_error) > np.deg2rad(100):
    #         linear_vel = 0.0
    #         # Gira con una velocidad angular constante y decidida
    #         angular_vel = 0.8 * np.sign(angle_error)
    #     else:
    #         # CASO 2: El giro es manejable.
    #         # La velocidad angular sigue siendo proporcional al error.
    #         angular_vel = 2.0 * angle_error

    #         # La velocidad lineal se reduce linealmente, no con coseno.
    #         # Esto es mucho menos agresivo.
    #         # Cuando el error es 0, el factor es 1.0. Cuando el error es 100 grados, el factor es 0.
    #         # Permite avanzar lentamente mientras se gira.
    #         scale_factor = 1.0 - (abs(angle_error) / np.deg2rad(100))
    #         linear_vel = self.linear_velocity * scale_factor

    #     # Limitar las velocidades finales
    #     angular_vel = float(np.clip(angular_vel, -1.2, 1.2))
    #     linear_vel = float(np.clip(linear_vel, 0.0, self.linear_velocity))

    #     # Publicar el comando
    #     twist_msg = Twist()
    #     twist_msg.linear.x = linear_vel
    #     twist_msg.angular.z = angular_vel
    #     self.cmd_vel_pub.publish(twist_msg)

    # def follow_path(self, estimated_pose):
    #     """
    #     Controlador de seguimiento de ruta (Pure Pursuit) mejorado para ser más seguro en las curvas.
    #     """
    #     robot_pos = np.array([estimated_pose.position.x, estimated_pose.position.y])
    #     _, _, robot_yaw = quaternion_to_euler(
    #         estimated_pose.orientation.x, 
    #         estimated_pose.orientation.y, 
    #         estimated_pose.orientation.z, 
    #         estimated_pose.orientation.w
    #     )

    #     if self.current_path is None or self.current_path_index >= len(self.current_path):
    #         self.get_logger().info("Follow_path: No hay ruta o se ha completado.")
    #         self.stop_robot()
    #         self.state = State.IDLE
    #         return

    #     # Comprobar si hemos llegado al objetivo final
    #     goal_pos = np.array(self.current_path[-1])
    #     if np.linalg.norm(robot_pos - goal_pos) < self.goal_tolerance:
    #         self.get_logger().info("¡Objetivo alcanzado!")
    #         self.state = State.IDLE
    #         self.stop_robot()
    #         return
        
    #     # --- Búsqueda del punto de mira (lookahead_point) ---
    #     lookahead_point = None
    #     # Avanzar el índice del path para no quedarse atascado en puntos ya pasados
    #     for i in range(self.current_path_index, len(self.current_path)):
    #         path_point = np.array(self.current_path[i])
    #         dist = np.linalg.norm(robot_pos - path_point)
    #         if dist > self.lookahead_distance:
    #             lookahead_point = path_point
    #             self.current_path_index = i # Actualizamos el índice
    #             break
        
    #     # Si no encontramos un punto de mira (porque estamos cerca del final), usamos el último punto
    #     if lookahead_point is None:
    #         lookahead_point = goal_pos

    #     # --- Lógica de control ---
    #     # Calcular el ángulo hacia el punto de mira
    #     angle_to_target = np.arctan2(lookahead_point[1] - robot_pos[1], lookahead_point[0] - robot_pos[0])
    #     angle_error = angle_to_target - robot_yaw
    #     # Normalizar el error de ángulo a [-pi, pi]
    #     angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
    #     # ** MODIFICACIÓN CLAVE: Control de velocidad adaptativo **
    #     # La velocidad angular es proporcional al error de ángulo.
    #     # Una ganancia de 2.0 suele funcionar bien.
    #     angular_vel = 2.0 * angle_error
        
    #     # La velocidad lineal se reduce drásticamente cuando se necesita girar.
    #     # Usamos cos(angle_error) que es 1 cuando el error es 0 (recto) y 0 cuando el error es 90 grados.
    #     # Esto hace que el robot casi se detenga para hacer giros cerrados.
    #     linear_vel = self.linear_velocity * max(0, np.cos(angle_error))

    #     # Limitar las velocidades a un máximo
    #     angular_vel = float(np.clip(angular_vel, -1.2, 1.2)) # Permitir un poco más de vel. angular
    #     linear_vel = float(np.clip(linear_vel, 0.0, self.linear_velocity))

    #     # Publicar el comando
    #     twist_msg = Twist()
    #     twist_msg.linear.x = linear_vel
    #     twist_msg.angular.z = angular_vel
    #     self.cmd_vel_pub.publish(twist_msg)


    # def follow_path(self, estimated_pose):
    #     robot_pos = np.array([estimated_pose.position.x, estimated_pose.position.y])
    #     _, _, robot_yaw = quaternion_to_euler(
    #         estimated_pose.orientation.x, 
    #         estimated_pose.orientation.y, 
    #         estimated_pose.orientation.z, 
    #         estimated_pose.orientation.w
    #     )

    #     if self.current_path is None or len(self.current_path) == 0:
    #         self.stop_robot()
    #         self.state = State.IDLE
    #         return

    #     goal_pos = np.array(self.current_path[-1])
    #     if np.linalg.norm(robot_pos - goal_pos) < self.goal_tolerance:
    #         self.get_logger().info("¡Objetivo alcanzado!")
    #         self.state = State.IDLE
    #         self.stop_robot()
    #         return
        
    #     # SOLUCIÓN 1: Buscar punto más cercano en el path
    #     min_dist = float('inf')
    #     closest_index = self.current_path_index
        
    #     for i in range(max(0, self.current_path_index - 2), len(self.current_path)):
    #         path_point = np.array(self.current_path[i])
    #         dist = np.linalg.norm(robot_pos - path_point)
    #         if dist < min_dist:
    #             min_dist = dist
    #             closest_index = i
        
    #     self.current_path_index = max(self.current_path_index, closest_index)
        
    #     # SOLUCIÓN 2: Usar punto lookahead más inteligente
    #     lookahead_point = None
    #     target_distance = max(self.lookahead_distance, 0.2)  # Mínimo 20cm
        
    #     for i in range(self.current_path_index, len(self.current_path)):
    #         path_point = np.array(self.current_path[i])
    #         dist_to_point = np.linalg.norm(robot_pos - path_point)
            
    #         if dist_to_point >= target_distance:
    #             lookahead_point = path_point
    #             break
        
    #     # Si no hay punto lookahead, usar el siguiente punto en el path
    #     if lookahead_point is None:
    #         if self.current_path_index + 1 < len(self.current_path):
    #             lookahead_point = np.array(self.current_path[self.current_path_index + 1])
    #         else:
    #             lookahead_point = goal_pos
        
    #     # SOLUCIÓN 3: Control mejorado con zona muerta angular
    #     angle_to_target = np.arctan2(
    #         lookahead_point[1] - robot_pos[1], 
    #         lookahead_point[0] - robot_pos[0]
    #     )
    #     angle_error = angle_to_target - robot_yaw
    #     angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
    #     # ZONA MUERTA ANGULAR - Evita oscilaciones
    #     angle_threshold = np.deg2rad(10)  # 10 grados de zona muerta
        
    #     if abs(angle_error) < angle_threshold:
    #         # Si el ángulo es pequeño, ir hacia adelante con corrección mínima
    #         angular_vel = 0.5 * angle_error  # Ganancia muy baja
    #         linear_vel = self.linear_velocity
    #     elif abs(angle_error) > np.deg2rad(90):  # Si necesita girar mucho
    #         # Rotar en el lugar sin moverse hacia adelante
    #         angular_vel = 0.8 * np.sign(angle_error)  # Velocidad angular constante
    #         linear_vel = 0.0
    #     else:
    #         # Control proporcional normal
    #         angular_vel = 1.0 * angle_error
    #         linear_vel = self.linear_velocity * (1.0 - abs(angle_error) / np.pi)
        
    #     # Limitar velocidades
    #     angular_vel = float(np.clip(angular_vel, -1.0, 1.0))
    #     linear_vel = float(np.clip(linear_vel, 0.0, self.linear_velocity))
        
    #     # SOLUCIÓN 4: Evitar comandos de velocidad muy pequeños
    #     if abs(angular_vel) < 0.05:
    #         angular_vel = 0.0
    #     if linear_vel < 0.02:
    #         linear_vel = 0.0
        
    #     twist_msg = Twist()
    #     twist_msg.linear.x = linear_vel
    #     twist_msg.angular.z = angular_vel
    #     self.cmd_vel_pub.publish(twist_msg)