import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, PoseWithCovarianceStamped, TransformStamped, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from scipy import ndimage
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
import heapq
from enum import Enum
import random

class AStarNode:
    def __init__(self, position, parent=None):
        self.parent = parent
        self.position = position # Tupla (x, y) en coordenadas de la grilla
        self.g = 0 # Costo desde el inicio hasta el nodo actual
        self.h = 0 # Heurística: costo estimado desde el nodo actual hasta el final
        self.f = 0 # Costo total (g + h)

    def __eq__(self, other):
        return self.position == other.position

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
        self.declare_parameter('obstacle_detection_distance', 0.3) # Distancia minima que considero que hay obstaculo inminente
        self.declare_parameter('obstacle_avoidance_turn_speed', 0.2) # Velocidad angular usada cuando detectas obstáculo y entras en modo evasión

        # --- Parameters to set ---
        # TODO: Setear valores default
        self.declare_parameter('num_particles', 20)
        # Alpha Modelo de odometria
        self.declare_parameter('alpha1', 0.005)  # Error rotacional debido a rotación
        self.declare_parameter('alpha2', 0.005)  # Error rotacional debido a traslación
        self.declare_parameter('alpha3', 0.001)  # Error traslacional debido a traslación
        self.declare_parameter('alpha4', 0.001)  # Error traslacional debido a rotación
        self.declare_parameter('z_hit', 0.95)  # Peso del modelo de hit (peso de la probabilidad de que la lectura coincida con el mapa)
        self.declare_parameter('z_rand', 0.1)  # Peso de detección aleatoria
        self.declare_parameter('lookahead_distance', 0.2)  # Reducir de 0.7 a 0.3
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('linear_velocity', 0.2) # Velocidad lineal que asignas al avanzar en la rama “avanzar con giro moderado”.
        self.declare_parameter('path_pruning_distance', 0.15)  # Distancia para podar path
        self.declare_parameter('safety_margin_cells', 5)  # Margen de seguridad en celdas
    
        self.declare_parameter('yaw_tolerance', 0.05)
        self.declare_parameter('kp_ang', 0.7) 
        self.declare_parameter('max_ang_speed', 1)


        self.kp_ang = self.get_parameter('kp_ang').value
        self.max_ang_speed = self.get_parameter('max_ang_speed').value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
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

        self.current_path_index = 0
        self.initial_convergence_done = False
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
        self.get_logger().warn('MyPyAMCL node initialized.')

        self.cant_index = None

    def map_callback(self, msg):
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            self.grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
            self.inflate_map()
            self.get_logger().warn('Map and inflated map processed.')

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
        self.get_logger().warn(f"New goal received: ({self.goal_pose.position.x:.2f}, {self.goal_pose.position.y:.2f}). State -> PLANNING")
        self.state = State.PLANNING
        self.current_path = None

    def initial_pose_callback(self, msg):
        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Initial pose frame is '{msg.header.frame_id}' but expected '{self.map_frame_id}'. Ignoring.")
            return
        self.get_logger().warn('Initial pose received.')
        self.initialize_particles(msg.pose.pose)
        self.initial_pose_received = True
        self.last_odom_pose = None # Reset odom tracking

    def initialize_particles(self, initial_pose):
        # TODO: Inicializar particulas en base a la pose inicial con variaciones aleatorias
        # Deben ser la misma cantidad de particulas que self.num_particles
        # Deben tener un peso
        for idx in range(self.num_particles):
            orientation = initial_pose.orientation
            poseX = initial_pose.position.x + np.random.normal(0,0.1)
            poseY = initial_pose.position.y + np.random.normal(0,0.1)
            _, _, theta = quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)
            self.particles[idx] = np.array([poseX, poseY, theta + np.random.normal(0,0.1)])
        
            gx, gy = self.world_to_grid(poseX, poseY)
            if gx < 0 or gx >= self.map_data.info.width or gy < 0 or gy >= self.map_data.info.height or self.grid[gy, gx] > 50:
                self.weights[idx] = 1e-10  # Peso muy bajo si está en obstáculo
            else:
                self.weights[idx] = 1.0 / self.num_particles

        self.weights /= np.sum(self.weights)  # Normalizar pesos
        self.publish_particles()

    def initialize_particles_randomly(self):
        # TODO: Inizializar particulas aleatoriamente en todo el mapa
        
        free_space_gy, free_space_gx = np.where(self.grid == 0)

        random_indices = np.random.choice(len(free_space_gy), self.num_particles)

        particle_gx = free_space_gx[random_indices]
        particle_gy = free_space_gy[random_indices]

        wx, wy = self.grid_to_world(particle_gx, particle_gy)
        w_theta = np.random.uniform(-np.pi, np.pi, self.num_particles)
        self.particles[:, 0] = wx + np.random.normal(0, 0.2, self.num_particles)
        self.particles[:, 1] = wy + np.random.normal(0, 0.2, self.num_particles)
        self.particles[:, 2] = w_theta

        # Inicialmente, todas las partículas tienen el mismo peso.
        self.weights.fill(1.0 / self.num_particles)

        self.publish_particles()

    def stop_robot(self):
        """Publica un mensaje Twist con todas las velocidades a cero."""
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)

    def check_for_imminent_obstacle(self):
        """Verifica el escaneo láser en un cono frontal para detectar obstáculos cercanos."""
        if self.latest_scan is None:
            return False

        # Defino el cono de detección frontal (ej. +/- 30 grados)
        front_angle_range = np.deg2rad(20)
        
        for i, dist in enumerate(self.latest_scan.ranges):
            angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
            if abs(angle) < front_angle_range / 2.0:
                if 0 < dist < self.obstacle_detection_distance:
                    self.get_logger().warn(f"Obstaculo al frente")
                    return True
        return False

    def create_path_message(self, grid_path):
        """Convierte path en coordenadas de grilla a mensaje Path de ROS."""
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame_id
        now = self.get_clock().now().to_msg()
        path_msg.header.stamp = now
        for grid_point in grid_path:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = now
            pose_stamped.header.frame_id = self.map_frame_id
            pose_stamped.pose.position.x = grid_point[0]
            pose_stamped.pose.position.y = grid_point[1]
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0  # Sin rotación específica
            
            path_msg.poses.append(pose_stamped)
        
        return path_msg

    def force_initial_convergence(self):
        """Fuerza convergencia inicial aplicando múltiples iteraciones del modelo de medición"""
        if self.latest_scan is None:
            return
        
        # self.get_logger().info("Forzando convergencia inicial...")
        
        # Aplicar modelo de medición y resampleo múltiples veces
        for iteration in range(7):  # 5 iteraciones
            self.measurement_model()
            
            # Verificar si hay convergencia
            effective_particles = 1.0 / np.sum(self.weights ** 2)
            self.get_logger().warn(f"Iteración {iteration + 1}: Partículas efectivas = {effective_particles:.1f}")
            
            if effective_particles < self.num_particles * 0.5:  # Si menos del 50% son efectivas
                self.resample()
            
            # Si la mayoría del peso está en pocas partículas, consideramos convergencia
            max_weight = np.max(self.weights)
            if max_weight > 0.1:  # Si una partícula tiene más del 10% del peso
                self.get_logger().warn(f"Convergencia detectada (peso máximo: {max_weight:.3f})")
                break
        
        self.publish_particles()

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

        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            # Acá se detiene el robot si se pierde la odometría
            if self.state in [State.NAVIGATING, State.AVOIDING_OBSTACLE]:
                self.stop_robot()
            return

        # Actualizo la pose de las particulas
        # if self.last_odom_pose is not None:
        #     self.get_logger().warn(f"Hace el motion model")
        self.motion_model(current_odom_tf)

        self.measurement_model()

        self.resample()

        estimated_pose = self.estimate_pose()
        self.get_logger().warn(f"[DEBUG] Pose estimada: x={estimated_pose.position.x:.2f}, y={estimated_pose.position.y:.2f}")
        # self.get_logger().warn(f"Particulas: {self.particles}")

        if self.state == State.PLANNING:
            path = self.A_algorithm(estimated_pose, self.goal_pose)

            if path:
                self.get_logger().warn(f'Ruta encontrada: {path}. State -> NAVIGATING')
                self.current_path = path
                self.current_path_index = 0 
                self.publish_path(self.create_path_message(path))
                self.state = State.NAVIGATING
            else:
                self.get_logger().error('No se pudo encontrar una ruta al objetivo. State -> IDLE')
                self.state = State.IDLE
        
        # CASO 2: Tiene una ruta y la esta siguiendo
        elif self.state == State.NAVIGATING:
            # Acá verifico si hay un obstaculo inminente
            if self.check_for_imminent_obstacle():
                self.get_logger().warn('Obstáculo detectado! State -> AVOIDING_OBSTACLE')
                self.state = State.AVOIDING_OBSTACLE
                return
            
            # FASE DE ALINEACIÓN antes del Pure Pursuit
            robot_x = estimated_pose.position.x
            robot_y = estimated_pose.position.y
            _, _, robot_yaw = quaternion_to_euler(
                estimated_pose.orientation.x,
                estimated_pose.orientation.y,
                estimated_pose.orientation.z,
                estimated_pose.orientation.w
            )
            if np.hypot(self.goal_pose.position.x - robot_x, self.goal_pose.position.y - robot_y) < self.goal_tolerance:
                self.get_logger().warn(f"¡Objetivo alcanzado!")
                self.stop_robot()
                self.state = State.IDLE
                return

            target_index = self.search_target_point_index(robot_x, robot_y)
            self.current_path_index = target_index
            target_x, target_y = self.current_path[target_index]
            
            # Calculo el ángulo hacia el objetivo
            target_angle = np.arctan2(target_y - robot_y, target_x - robot_x)
            angle_error = target_angle - robot_yaw
            # Normalizo el ángulo al rango [-pi, pi]
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
            
            self.get_logger().warn(f"[DEBUG] Angle error: {np.degrees(angle_error):.1f}°, Angle: {angle_error}")

            if abs(angle_error) > self.yaw_tolerance:
                self.get_logger().warn(f"[ALIGN] Alineando robot - Error: {np.degrees(angle_error):.1f}°")
                angular_velocity = self.kp_ang * angle_error

                angular_velocity = np.sign(angular_velocity) * 0.15
                angular_velocity = np.clip(
                    angular_velocity,
                    -self.max_ang_speed,
                    self.max_ang_speed
                )
                twist_msg = Twist()
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = angular_velocity
                self.cmd_vel_pub.publish(twist_msg)
            else:
                self.get_logger().warn(f"[PURSUIT] Robot alineado - Siguiendo ruta")
                self.pure_pursuit_control(estimated_pose)

        elif self.state == State.AVOIDING_OBSTACLE:
            self.get_logger().warn(f"Dejó de girar")
            self.stop_robot()
            self.state = State.NAVIGATING


        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)

    def search_target_point_index(self, robot_x, robot_y):
        """
        Busca el índice del punto objetivo en la ruta usando lookahead distance
        """
        if self.current_path is None or len(self.current_path) == 0:
            return 0
        
        # Encontrar el punto más cercano en la ruta
        distances = []
        for i, point in enumerate(self.current_path):
            dist = np.sqrt((robot_x - point[0])**2 + (robot_y - point[1])**2)
            distances.append(dist)
        
        for i in range(self.current_path_index, len(self.current_path)):
            point = self.current_path[i]
            distance = np.sqrt((robot_x - point[0])**2 + (robot_y - point[1])**2)
            
            if distance >= self.lookahead_distance:
                return i
        
        # Si no se encuentra, retornar el último punto
        return len(self.current_path) - 1

    def pure_pursuit_control(self, estimated_pose):
        """
        Controlador de seguimiento de ruta usando Pure Pursuit
        """
        robot_x = estimated_pose.position.x
        robot_y = estimated_pose.position.y
        _, _, robot_yaw = quaternion_to_euler(
            estimated_pose.orientation.x,
            estimated_pose.orientation.y,
            estimated_pose.orientation.z,
            estimated_pose.orientation.w
        )

        # Buscar el punto objetivo usando la función auxiliar
        target_index = self.search_target_point_index(robot_x, robot_y)
        target_x, target_y = self.current_path[target_index]
        
        self.get_logger().warn(f"[DEBUG] Robot: ({robot_x:.2f}, {robot_y:.2f}), Target: ({target_x:.2f}, {target_y:.2f}), Index: {target_index}/{len(self.current_path)-1}")

        # Calcular ángulo hacia el objetivo (Pure Pursuit)
        alpha = np.arctan2(target_y - robot_y, target_x - robot_x) - robot_yaw
        # Normalizar el ángulo al rango [-pi, pi]
        # alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        # Calcular distancia al punto objetivo
        lookahead_actual = np.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)

        curvature = 2.0 * np.sin(alpha) / lookahead_actual
            
        omega = self.linear_velocity * curvature

        self.get_logger().warn(f"[DEBUG] Alpha: {omega:.1f}°, Lin: {self.linear_velocity:.2f}, Ang: {omega:.2f}")

        # Publicar comandos de velocidad
        twist_msg = Twist()
        twist_msg.linear.x = self.linear_velocity
        twist_msg.angular.z = omega
        self.cmd_vel_pub.publish(twist_msg)

    def get_heuristic(self, cell, goal):
        
        y_c, x_c = cell
        y_g, x_g = goal

        dy = y_c - y_g
        dx = x_c - x_g

        heuristic = np.hypot(dy, dx)

        return heuristic

    def A_algorithm(self, start_pose, goal_pose):
        """
        Planifica una ruta desde una pose de inicio a una pose objetivo usando el algoritmo A*.
        Basado en la implementación de planning_framework.py
        :param start_pose: Pose de inicio (geometry_msgs/Pose).
        :param goal_pose: Pose objetivo (geometry_msgs/Pose).
        :return: Una lista de tuplas (x, y) con las coordenadas del mundo si se encuentra ruta, de lo contrario None.
        """
        self.get_logger().warn("A* planning started.")
        
        # Convertir poses del mundo a coordenadas de grilla
        start_grid = self.world_to_grid(start_pose.position.x, start_pose.position.y)
        goal_grid = self.world_to_grid(goal_pose.position.x, goal_pose.position.y)
        
        # Validar que start y goal estén dentro del mapa
        map_h, map_w = self.inflated_grid.shape
        if not (0 <= start_grid[0] < map_w and 0 <= start_grid[1] < map_h):
            self.get_logger().error("La posición inicial está fuera de los límites del mapa")
            return None
        if not (0 <= goal_grid[0] < map_w and 0 <= goal_grid[1] < map_h):
            self.get_logger().error("La posición del objetivo está fuera de los límites del mapa")
            return None
        
        # Convertir formato de coordenadas para compatibilidad con planning_framework
        # planning_framework usa [y, x], nosotros usamos [x, y]
        start_pf = [start_grid[1], start_grid[0]]
        goal_pf = [goal_grid[1], goal_grid[0]]
        
        # Crear mapa de ocupación normalizado (0-1) desde inflated_grid
        occ_map = self.inflated_grid.astype(float) / 100.0
        occ_map = np.clip(occ_map, 0.0, 1.0)
        
        # Inicializar estructuras de datos
        costs = np.ones(occ_map.shape) * np.inf
        closed_flags = np.zeros(occ_map.shape, dtype=bool)
        predecessors = -np.ones(occ_map.shape + (2,), dtype=np.int64)
        
        # Calcular heurística para A*
        heuristic = np.zeros(occ_map.shape)
        heuristic_factor = 1
        for x in range(occ_map.shape[0]):
            for y in range(occ_map.shape[1]):
                heuristic[x, y] = self._get_heuristic([x, y], goal_pf) * heuristic_factor
        
        # Inicializar búsqueda
        parent = np.array(start_pf)
        costs[start_pf[0], start_pf[1]] = 0
        
        # Bucle principal de búsqueda
        while not np.array_equal(parent, goal_pf):
            # Costos de celdas candidatas para expansión (no en lista cerrada)
            open_costs = np.where(closed_flags, np.inf, costs) + heuristic
            
            # Encontrar celda con costo mínimo en lista abierta
            x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)
            
            # Romper bucle si costos mínimos son infinitos (no hay más celdas abiertas)
            if open_costs[x, y] == np.inf:
                break
            
            # Establecer como padre y ponerlo en lista cerrada
            parent = np.array([x, y])
            closed_flags[x, y] = True
            
            # Actualizar costos y predecesores para vecinos
            neighbors = self._get_neighborhood(parent, occ_map.shape)
            for neighbor in neighbors:
                y_n, x_n = neighbor
                
                # Calcular costo del borde
                edge_cost = self._get_edge_cost(parent, neighbor, occ_map)
                if edge_cost == float('inf'):
                    continue
                
                new_cost = costs[parent[0], parent[1]] + edge_cost
                
                if new_cost < costs[y_n, x_n]:
                    costs[y_n, x_n] = new_cost
                    predecessors[y_n, x_n] = parent
        
        # Reconstruir ruta desde goal hasta start
        if np.array_equal(parent, goal_pf):
            # Encontrar ruta desde goal hasta start
            grid_path = []
            current = np.array(goal_pf)
            
            while predecessors[current[0], current[1]][0] >= 0:
                # Convertir de formato [y, x] a [x, y] para world_to_grid
                grid_x = current[1]
                grid_y = current[0]
                world_coords = self.grid_to_world(grid_x, grid_y)
                grid_path.append(world_coords)
                current = predecessors[current[0], current[1]]
            
            # Agregar punto de inicio
            world_coords = self.grid_to_world(start_grid[0], start_grid[1])
            grid_path.append(world_coords)
            
            # Invertir para que vaya del inicio al fin
            grid_path = grid_path[::-1]
            
            self.get_logger().warn(f"A* encontrao ruta con {len(grid_path)} celdas.")
            # por distancia recorrida y costo por pasar cerca de los obstaculos 
            self.get_logger().warn(f"Costo acumulado del camino: {costs[goal_pf[0], goal_pf[1]]:.2f}")
            
            return grid_path
        else:
            self.get_logger().warn("A* failed to find a path.")
            return None

    # Obtiene los vecinos más cercanos
    def _get_neighborhood(self, cell, occ_map_shape):
        fila, col = cell
        filas_totales, columnas_totales = occ_map_shape
        
        movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        vecinos = []
        for df, dc in movimientos:
            nueva_fila, nueva_col = fila + df, col + dc
            if 0 <= nueva_fila < filas_totales and 0 <= nueva_col < columnas_totales:
                vecinos.append((nueva_fila, nueva_col))
        
        return vecinos

    # Calcula el costo de moverse al padre
    def _get_edge_cost(self, parent, child, occ_map):
        y_p, x_p = parent
        y_c, x_c = child
        p_ocup = occ_map[y_c, x_c]
        
        if p_ocup > 0.7:
            return float('inf')
        
        p_p = np.array([x_p, y_p])
        p_c = np.array([x_c, y_c])
        distancia = np.linalg.norm(p_c - p_p)
        
        # Costo del borde considerando probabilidad de ocupación
        edge_cost = distancia * (1.0 + float(p_ocup))
        
        return edge_cost

    def _get_heuristic(self, cell, goal):
        y_c, x_c = cell
        y_g, x_g = goal
        
        dy = y_c - y_g
        dx = x_c - x_g
        
        # Distancia euclidiana como heurística
        heuristic = np.hypot(dy, dx)
        
        return heuristic
    
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
        
        if self.last_odom_pose is None:
            self.last_odom_pose = current_odom_pose
            return
        
        # Extraer posiciones y orientaciones
        x_prev = self.last_odom_pose.translation.x
        y_prev = self.last_odom_pose.translation.y
        _, _, theta_prev = quaternion_to_euler(
            self.last_odom_pose.rotation.x,
            self.last_odom_pose.rotation.y, 
            self.last_odom_pose.rotation.z,
            self.last_odom_pose.rotation.w
        )
        
        x_curr = current_odom_pose.translation.x
        y_curr = current_odom_pose.translation.y
        _, _, theta_curr = quaternion_to_euler(
            current_odom_pose.rotation.x,
            current_odom_pose.rotation.y,
            current_odom_pose.rotation.z, 
            current_odom_pose.rotation.w
        )
        
        delta_x = x_curr - x_prev
        delta_y = y_curr - y_prev
        delta_theta = theta_curr - theta_prev

        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
        
        delta_trans = np.sqrt(delta_x**2 + delta_y**2)
        delta_rot1 = np.arctan2(delta_y, delta_x) - theta_prev
        delta_rot2 = delta_theta - delta_rot1
        
        delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
        delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))
        
        # Apply motion model to each particle
        for i in range(self.num_particles):
            # Add noise based on motion
            sigma_rot1 = self.alphas[0] * abs(delta_rot1) + self.alphas[1] * abs(delta_trans)
            delta_rot1_noisy = delta_rot1 + random.gauss(0, sigma_rot1)
            
            sigma_trans = self.alphas[2] * abs(delta_trans) + self.alphas[3] * abs(delta_rot1 + delta_rot2)
            translation_noisy = delta_trans + random.gauss(0, sigma_trans)
            
            sigma_delta_rot2 = self.alphas[0] * abs(delta_rot2) + self.alphas[1] * abs(delta_trans)
            delta_ro2_noisy = delta_rot2 + random.gauss(0, sigma_delta_rot2)

            # Update particle pose
            self.particles[i, 0] += translation_noisy * np.cos(self.particles[i, 2] + delta_rot1_noisy)
            self.particles[i, 1] += translation_noisy * np.sin(self.particles[i, 2] + delta_rot1_noisy)
            self.particles[i, 2] += delta_rot1_noisy + delta_ro2_noisy
            
        self.last_odom_pose = current_odom_pose

    def measurement_model(self):
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
        
        for i in range(self.num_particles):
            particle_x, particle_y, particle_yaw = self.particles[i]
            
            prob = 1.0
            
            step = max(1, len(scan_ranges) // 100)  # Sample ~50 rays
            
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

    # Bien
    def estimate_pose(self):
        # Normalizar pesos para asegurar que sumen 1
        normalized_weights = self.weights / np.sum(self.weights)
        
        # Mejor Particula
        best_particle_idx = np.argmax(normalized_weights)
        x_best = self.particles[best_particle_idx, 0]
        y_best = self.particles[best_particle_idx, 1]
        theta_best = self.particles[best_particle_idx, 2]
        
        # Si la dispersión es alta, usar la mejor partícula; si es baja, usar promedio
        x_est = x_best
        y_est = y_best
        theta_est = theta_best
        
        # Crear mensaje de pose
        pose = Pose()
        pose.position.x = x_est
        pose.position.y = y_est
        pose.position.z = 0.0
        
        # Convertir ángulo a quaternion
        q = R.from_euler('z', theta_est).as_quat()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
                
        return pose

    def publish_pose(self, estimated_pose):
        """
        Publica la pose estimada en el tópico 'amcl_pose' como PoseWithCovarianceStamped.
        """
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.map_frame_id
        pose_msg.pose.pose = estimated_pose
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

        # Extraer traslación
        t.transform.translation.x = map_to_odom_mat[0, 3]
        t.transform.translation.y = map_to_odom_mat[1, 3]
        t.transform.translation.z = map_to_odom_mat[2, 3]
        
        # Extraer rotación
        r = R.from_matrix(map_to_odom_mat[:3, :3])
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
        
        self.get_logger().warn(f"Mapa inflado con margen de {self.safety_margin_cells} celdas")

def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 