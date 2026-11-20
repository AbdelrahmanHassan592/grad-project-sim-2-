"""Navigate to a selected course marker using filtered odometry and live LiDAR."""
from collections import deque
from heapq import heappop, heappush
from math import atan2, ceil, cos, hypot, isfinite, sin, tau
from time import monotonic

import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from my_car_description_2.landmark_detection import beacon_offset

RESOLUTION = 0.20
CLEARANCE = 0.55
MARKER_COLORS = {'A': (0.12, 0.75, 0.36), 'B': (0.15, 0.48, 1.0), 'C': (1.0, 0.5, 0.17)}
GOAL_TOLERANCE = 0.22
LIDAR_OFFSET_X = 0.15  # lidar_link is 0.15 m forward of base_link.
MAX_DATA_AGE = 0.6
MAP_PERIOD = 0.25
INFLATION = [
    (x, y)
    for x in range(-3, 4) for y in range(-3, 4)
    if hypot(x, y) * RESOLUTION <= CLEARANCE + RESOLUTION / 2
]
NEIGHBORS = [(dx, dy, hypot(dx, dy)) for dx in (-1, 0, 1)
             for dy in (-1, 0, 1) if dx or dy]


def grid_cell(x, y):
    return round(x / RESOLUTION), round(y / RESOLUTION)


def search(start, goal, free, blocked):
    """Return an A* route through observed free space."""
    if goal not in free or goal in blocked:
        return []
    queue = [(0, 0, start)]
    costs = {start: 0}
    parents = {}
    while queue:
        _, cost, cell = heappop(queue)
        if cost != costs[cell]:
            continue
        if cell == goal:
            path = [cell]
            while cell != start:
                cell = parents[cell]
                path.append(cell)
            return list(reversed(path))
        x, y = cell
        for dx, dy, step_cost in NEIGHBORS:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            next_cost = cost + step_cost
            if neighbor not in free or neighbor in blocked:
                continue
            if dx and dy and any(side in blocked or side not in free for side in ((nx, y), (x, ny))):
                continue
            if next_cost >= costs.get(neighbor, float('inf')):
                continue
            costs[neighbor] = next_cost
            parents[neighbor] = cell
            distance = hypot(nx - goal[0], ny - goal[1])
            heappush(queue, (next_cost + distance, next_cost, neighbor))
    return []


def next_viewpoint(start, free, blocked, visited, toward=None):
    """Choose a reachable, unsearched observation point in the learned map."""
    queue = deque([(start, 0)])
    seen = {start}
    best = None
    best_score = float('-inf')
    while queue:
        (x, y), cost = queue.popleft()
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            cell = (x + dx, y + dy)
            if cell not in seen and cell in free and cell not in blocked:
                seen.add(cell)
                queue.append((cell, cost + 1))
        if x % 3 or y % 3 or cost < 8:
            continue
        px, py = x * RESOLUTION, y * RESOLUTION
        spacing = min(hypot(px - vx, py - vy) for vx, vy in visited)
        if spacing < 1.45:
            continue
        frontier = sum((x + dx, y + dy) not in free and (x + dx, y + dy) not in blocked
                       for dx in (-5, -3, 0, 3, 5) for dy in (-5, -3, 0, 3, 5))
        score = min(spacing, 4.0) + 0.13 * frontier - 0.065 * cost
        if toward is not None:
            score -= 0.7 * hypot(px - toward[0], py - toward[1])
        if score > best_score:
            best, best_score = (px, py), score
    return best


def segment_clear(x, y, target, blocked):
    dx, dy = target[0] - x, target[1] - y
    samples = max(1, ceil(hypot(dx, dy) / (RESOLUTION / 4)))
    origin = grid_cell(x, y)
    for i in range(1, samples + 1):
        sample = grid_cell(x + dx * i / samples, y + dy * i / samples)
        if sample != origin and sample in blocked:
            return False
    return True


class AutonomousDrive(Node):
    def __init__(self):
        super().__init__('drive_to_goal')
        self.pose = None
        self.pose_history = deque(maxlen=90)
        self.points = []
        self.free = set()
        self.evidence = {}
        self.occupied = set()
        self.blocked = set()
        self.last_map = self.last_image = float('-inf')
        self.landmarks = {'A': (0.0, 0.0)}
        self.sightings = deque(maxlen=3)
        self.search_goal = None
        self.following_landmark = False
        self.searched = []
        self.swept = 0.0
        self.last_sweep_yaw = None
        self.pose_time = self.scan_time = float('-inf')
        self.location = 'A'
        self.destination = None
        self.target = None
        self.reported_detour = False
        self.velocity = self.create_publisher(Twist, '/cmd_vel', 10)
        self.markers = self.create_publisher(
            MarkerArray, '/course/markers', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.map_publisher = self.create_publisher(
            OccupancyGrid, '/course/map', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, '/odometry/filtered', self.read_pose, 10)
        self.create_subscription(LaserScan, '/scan', self.read_scan, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.read_image, qos_profile_sensor_data)
        self.create_subscription(String, '/course/destination', self.select_destination, 10)
        self.create_timer(0.10, self.drive, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.show_course()
        self.get_logger().info('At point A. Waiting for a destination.')

    def select_destination(self, data):
        selected = data.data.strip().upper()
        if selected not in MARKER_COLORS:
            self.get_logger().warning('Choose A, B, or C.')
        elif self.destination:
            self.get_logger().info(f'Already driving to point {self.destination}.')
        elif selected == self.location:
            self.get_logger().info(f'Already at point {selected}.')
        else:
            self.destination = selected
            self.target = None
            self.reported_detour = False
            self.sightings.clear()
            self.searched.clear()
            self.search_goal = None
            self.following_landmark = False
            self.last_sweep_yaw = None
            self.swept = 0.0
            if selected in self.landmarks:
                self.get_logger().info(f'Using the learned map to drive from {self.location} to {selected}.')
            else:
                self.get_logger().info(f'Searching for point {selected}. Its position is unknown.')

    def show_course(self, path=()):
        markers = []
        for name, (x, y) in self.landmarks.items():
            index = ord(name) - ord('A')
            disc = Marker(ns='destination', id=index, type=Marker.CYLINDER)
            disc.header.frame_id = 'odom'
            disc.pose.orientation.w = 1.0
            disc.pose.position = Point(x=x, y=y, z=-0.04)
            disc.scale.x = disc.scale.y = 1.26
            disc.scale.z = 0.012
            disc.color.r, disc.color.g, disc.color.b = MARKER_COLORS[name]
            disc.color.a = 0.60
            label = Marker(ns='label', id=index, type=Marker.TEXT_VIEW_FACING)
            label.header.frame_id = 'odom'
            label.pose.orientation.w = 1.0
            label.pose.position = Point(x=x, y=y, z=0.4)
            label.scale.z = 0.42
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = name
            markers.extend((disc, label))
        route = Marker(ns='route', id=0, type=Marker.LINE_STRIP)
        route.header.frame_id = 'odom'
        route.pose.orientation.w = 1.0
        route.scale.x = 0.045
        route.color.r, route.color.g, route.color.b, route.color.a = (0.13, 0.82, 1.0, 0.90)
        route.points = [Point(x=x * RESOLUTION, y=y * RESOLUTION, z=0.025) for x, y in path]
        route.action = Marker.ADD if route.points else Marker.DELETE
        markers.append(route)
        self.markers.publish(MarkerArray(markers=markers))

    def read_pose(self, data):
        position = data.pose.pose.position
        q = data.pose.pose.orientation
        yaw = atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if all(map(isfinite, (position.x, position.y, yaw))):
            self.pose = position.x, position.y, yaw
            self.pose_time = monotonic()
            stamp = data.header.stamp
            self.pose_history.append((stamp.sec + stamp.nanosec * 1e-9, self.pose))

    def read_image(self, image):
        now = monotonic()
        if self.destination not in ('B', 'C') or not self.pose_history or now - self.last_image < 0.12:
            return
        self.last_image = now
        offset = beacon_offset(image, self.destination)
        if offset is None:
            self.sightings.clear()
            return
        pose = self.pose_at(image.header.stamp)
        if pose is None:
            return
        x, y, yaw = pose
        forward, left = offset
        point = (x + cos(yaw) * forward - sin(yaw) * left,
                 y + sin(yaw) * forward + cos(yaw) * left)
        if self.sightings and hypot(point[0] - self.sightings[-1][0],
                                    point[1] - self.sightings[-1][1]) > 0.38:
            self.sightings.clear()
        self.sightings.append(point)
        if len(self.sightings) != self.sightings.maxlen:
            return
        average = tuple(sum(coords) / len(self.sightings) for coords in zip(*self.sightings))
        known = self.landmarks.get(self.destination)
        if known is None:
            self.landmarks[self.destination] = average
            self.target = None
            self.get_logger().info(
                f'Found point {self.destination} with the camera at ({average[0]:.2f}, {average[1]:.2f}) m.')
            self.show_course()
        elif forward < 3.0 and hypot(average[0] - known[0], average[1] - known[1]) < 0.45:
            self.landmarks[self.destination] = tuple(.85 * old + .15 * new for old, new in zip(known, average))

    def pose_at(self, stamp):
        time = stamp.sec + stamp.nanosec * 1e-9
        pose_stamp, pose = min(self.pose_history, key=lambda sample: abs(sample[0] - time))
        return pose if abs(pose_stamp - time) <= 0.15 else None

    def read_scan(self, data):
        # An infinite range is a valid clear ray. Reject scans with no usable rays.
        if not any(distance == float('inf') or data.range_min <= distance <= data.range_max
                   for distance in data.ranges):
            return
        self.points = []
        rays = []
        for index, distance in enumerate(data.ranges):
            if data.range_min <= distance <= data.range_max:
                angle = data.angle_min + index * data.angle_increment
                self.points.append((LIDAR_OFFSET_X + distance * cos(angle), distance * sin(angle)))
                rays.append((angle, distance, True))
            elif distance == float('inf'):
                rays.append((data.angle_min + index * data.angle_increment, data.range_max, False))
        self.scan_time = monotonic()
        if self.pose_history and self.scan_time - self.last_map >= MAP_PERIOD:
            pose = self.pose_at(data.header.stamp)
            if pose is not None:
                self.update_map(rays, pose)
                self.last_map = self.scan_time

    def update_map(self, rays, pose):
        x, y, yaw = pose
        sx, sy = x + LIDAR_OFFSET_X * cos(yaw), y + LIDAR_OFFSET_X * sin(yaw)
        hits = set()
        clear = {grid_cell(x, y)}
        for angle, length, hit in rays[::2]:
            c, s = cos(angle + yaw), sin(angle + yaw)
            for index in range(1, ceil(length / 0.13)):
                radius = index * 0.13
                clear.add(grid_cell(sx + radius * c, sy + radius * s))
            if hit:
                hits.add(grid_cell(sx + length * c, sy + length * s))
        self.free.update(clear - hits)
        for cell in clear - hits:
            self.evidence[cell] = max(-3, self.evidence.get(cell, 0) - 1)
        for cell in hits:
            self.evidence[cell] = min(5, self.evidence.get(cell, 0) + 2)
        self.occupied = {cell for cell, evidence in self.evidence.items() if evidence >= 2}
        self.blocked = {(gx + dx, gy + dy) for gx, gy in self.occupied | hits for dx, dy in INFLATION}
        cells = self.free | self.occupied
        x0, x1 = min(gx for gx, _ in cells), max(gx for gx, _ in cells)
        y0, y1 = min(gy for _, gy in cells), max(gy for _, gy in cells)
        message = OccupancyGrid()
        message.header.frame_id = 'odom'
        message.header.stamp = self.get_clock().now().to_msg()
        message.info.resolution = RESOLUTION
        message.info.width = x1 - x0 + 1
        message.info.height = y1 - y0 + 1
        message.info.origin.position = Point(x=(x0 - 0.5) * RESOLUTION, y=(y0 - 0.5) * RESOLUTION, z=-0.065)
        message.info.origin.orientation.w = 1.0
        message.data = [100 if (gx, gy) in self.occupied else 0 if (gx, gy) in self.free else -1
                        for gy in range(y0, y1 + 1) for gx in range(x0, x1 + 1)]
        self.map_publisher.publish(message)

    def drive(self):
        command = Twist()
        now = monotonic()
        if self.destination is None or now - min(self.pose_time, self.scan_time) > MAX_DATA_AGE:
            self.velocity.publish(command)
            return
        x, y, yaw = self.pose
        landmark = self.landmarks.get(self.destination)
        start = grid_cell(x, y)
        landmark_path = search(start, grid_cell(*landmark), self.free, self.blocked) if landmark else []
        exploring = not landmark_path
        if self.following_landmark != (not exploring):
            self.following_landmark = not exploring
            self.target = self.search_goal = None
            self.last_sweep_yaw = None
            self.swept = 0.0
            if exploring:
                self.show_course()
        goal = landmark
        if exploring:
            sweeping = self.search_goal is None or hypot(self.search_goal[0] - x, self.search_goal[1] - y) < 0.28
            if sweeping:
                if self.last_sweep_yaw is not None:
                    delta = yaw - self.last_sweep_yaw
                    self.swept += max(0.0, atan2(sin(delta), cos(delta)))
                self.last_sweep_yaw = yaw
                if self.swept < tau:
                    command.angular.z = 0.72
                    self.velocity.publish(command)
                    return
                self.searched.append((x, y))
                self.search_goal = next_viewpoint(start, self.free, self.blocked, self.searched, landmark)
                self.target = None
                self.last_sweep_yaw = None
                self.swept = 0.0
                if self.search_goal is None:
                    self.get_logger().warning(f'Could not reach point {self.destination} through the learned map.')
                    self.destination = None
                    self.velocity.publish(command)
                    return
            goal = self.search_goal
        remaining = hypot(goal[0] - x, goal[1] - y)
        if not exploring and remaining <= GOAL_TOLERANCE:
            self.location = self.destination
            self.destination = None
            self.target = None
            self.velocity.publish(command)
            self.show_course()
            self.get_logger().info(f'Reached point {self.location}. Filtered position: ({x:.2f}, {y:.2f}) m.')
            return

        blocked = self.blocked
        if (self.target is None or hypot(self.target[0] - x, self.target[1] - y) < 0.23
                or not segment_clear(x, y, self.target, blocked)):
            self.target = None
            path = search(start, grid_cell(*goal), self.free, blocked) if exploring else landmark_path
            if exploring and not path:
                self.searched.append(goal)
                self.search_goal = next_viewpoint(start, self.free, blocked, self.searched, landmark)
            if not exploring:
                self.show_course(path)
            if not exploring and path and not self.reported_detour and not segment_clear(x, y, goal, blocked):
                self.reported_detour = True
                self.get_logger().info(f'LiDAR detected a blocked route. Planning a detour to point {self.destination}.')
            # Keep a visible waypoint until reached or blocked. Replacing it on
            # every scan can make equally short routes alternate at a corner.
            for gx, gy in path[1:]:
                waypoint = gx * RESOLUTION, gy * RESOLUTION
                if hypot(waypoint[0] - x, waypoint[1] - y) > 1.0:
                    break
                if segment_clear(x, y, waypoint, blocked):
                    self.target = waypoint
        if self.target is not None:
            target_yaw = atan2(self.target[1] - y, self.target[0] - x)
            error = atan2(sin(target_yaw - yaw), cos(target_yaw - yaw))
            command.angular.z = max(-0.95, min(0.95, 2.0 * error))
            front_clear = all(not (0.0 < px < 0.55 and abs(py) < 0.30) for px, py in self.points)
            if front_clear and abs(error) < 0.50:
                command.linear.x = min(0.42, max(0.13, remaining)) * cos(error)
        self.velocity.publish(command)


def main():
    rclpy.init()
    node = AutonomousDrive()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.velocity.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()
