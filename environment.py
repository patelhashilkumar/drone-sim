import numpy as np
from typing import List, Tuple, Dict


class Environment:
    def __init__(self, width: int, height: int, obstacle_density: float = 0.2):
        self.width = width
        self.height = height
        self.max_altitude = 12  # Maximum flight altitude
        self.grid = np.zeros((height, width))  # Stores building heights (0 = ground)
        self.obstacle_density = obstacle_density
        self.dynamic_obstacles: List[Tuple[int, int]] = []
        self.drones: Dict[int, Dict] = {}
        self.start = None
        self.goal = None
        self.safety_radius = 3  # No obstacles within this radius of start/goal

    def add_drone(self, drone_id: int, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Add a drone with its start and goal positions"""
        if self._is_valid_position(*start) and self._is_valid_position(*goal):
            self.drones[drone_id] = {
                'start': start,
                'goal': goal,
                'current_position': start,
                'path': None
            }
            if self.start is None:
                self.start = start
            if self.goal is None:
                self.goal = goal
            return True
        return False

    def update_drone_position(self, drone_id: int, position: Tuple[int, int]):
        """Update the current position of a drone"""
        if drone_id in self.drones:
            self.drones[drone_id]['current_position'] = position

    def add_dynamic_obstacle(self, x: int, y: int, h: int = 4):
        """Add a dynamic obstacle that can appear during flight"""
        if self._is_valid_position(x, y) and not self._is_in_safety_zone(x, y):
            self.dynamic_obstacles.append((x, y))
            self.grid[y, x] = h
            return True
        return False

    def remove_dynamic_obstacle(self, x: int, y: int):
        """Remove a dynamic obstacle"""
        if (x, y) in self.dynamic_obstacles:
            self.dynamic_obstacles.remove((x, y))
            self.grid[y, x] = 0
            return True
        return False

    def check_path_validity(self, path: np.ndarray) -> bool:
        """Check if a path is still valid (no new obstacles)"""
        for i in range(len(path) - 1):
            points = self._interpolate_points(path[i], path[i + 1])
            for point in points:
                x, y = int(point[0]), int(point[1])
                if self.is_collision(x, y):
                    return False
        return True

    def _interpolate_points(self, p1: np.ndarray, p2: np.ndarray, steps: int = 10) -> np.ndarray:
        """Interpolate points between two waypoints"""
        return np.linspace(p1, p2, steps)

    def set_start(self, x: int, y: int) -> bool:
        """Set the start position"""
        if self._is_valid_position(x, y):
            self.start = (x, y)
            return True
        return False

    def set_goal(self, x: int, y: int) -> bool:
        """Set the goal position"""
        if self._is_valid_position(x, y):
            self.goal = (x, y)
            return True
        return False

    def _is_in_safety_zone(self, x: int, y: int) -> bool:
        """Check if a position is within the safety radius of any drone start/goal."""
        for drone_data in self.drones.values():
            for key in ('start', 'goal'):
                pos = drone_data[key]
                if abs(x - pos[0]) <= self.safety_radius and abs(y - pos[1]) <= self.safety_radius:
                    return True
        # Also check legacy start/goal
        if self.start and abs(x - self.start[0]) <= self.safety_radius and abs(y - self.start[1]) <= self.safety_radius:
            return True
        if self.goal and abs(x - self.goal[0]) <= self.safety_radius and abs(y - self.goal[1]) <= self.safety_radius:
            return True
        return False

    def generate_random_obstacles(self):
        """Generate clustered obstacles with safety zones around drone start/goal positions."""
        total_obstacles = int(self.width * self.height * self.obstacle_density)
        num_clusters = max(3, total_obstacles // 15)  # Each cluster ~15 cells
        obstacles_per_cluster = total_obstacles // num_clusters

        placed = 0
        for _ in range(num_clusters):
            # Pick a random cluster center
            cx = np.random.randint(2, self.width - 2)
            cy = np.random.randint(2, self.height - 2)

            # Skip if center is in a safety zone
            if self._is_in_safety_zone(cx, cy):
                continue

            for __ in range(obstacles_per_cluster):
                if placed >= total_obstacles:
                    break
                # Place obstacle near the cluster center with gaussian spread
                ox = int(np.clip(cx + np.random.normal(0, 2), 0, self.width - 1))
                oy = int(np.clip(cy + np.random.normal(0, 2), 0, self.height - 1))

                if not self._is_in_safety_zone(ox, oy):
                    self.grid[oy, ox] = 1
                    placed += 1

    def add_obstacle(self, x: int, y: int, h: int = 4) -> bool:
        """Add an obstacle at the specified position"""
        if self._is_valid_position(x, y):
            self.grid[y, x] = h
            return True
        return False

    def is_collision(self, x: int, y: int, z: int) -> bool:
        """Check if 3D position collides with an obstacle or exceeds altitude"""
        if not self._is_valid_position(x, y):
            return True # Out of bounds
        if z < 0 or z >= self.max_altitude:
            return True # Too high or below ground
        building_height = self.grid[y, x]
        return z <= building_height

    def _is_valid_position(self, x: int, y: int, *args) -> bool:
        """Check if position is within grid bounds"""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_obstacle_count(self) -> int:
        """Return total number of obstacle cells."""
        return int(np.sum(self.grid))

    def generate_city_blocks(self):
        """
        Generate structured city blocks instead of random noise.
        Creates rectangular obstacles representing buildings/blocks with clear corridors (streets).
        Drones start and end positions will be protected by the safety zone.
        """
        self.grid = np.zeros((self.height, self.width))
        
        # City configuration
        block_width_min, block_width_max = 4, 8
        block_height_min, block_height_max = 4, 8
        street_width = 3

        y = street_width
        while y < self.height - street_width:
            x = street_width
            block_h = np.random.randint(block_height_min, block_height_max)
            
            while x < self.width - street_width:
                block_w = np.random.randint(block_width_min, block_width_max)
                
                # Check bounds
                if x + block_w >= self.width:
                    block_w = self.width - x - 1
                if y + block_h >= self.height:
                    block_h = self.height - y - 1

                # Try to place the block if it's not entirely inside a safety zone
                safe_to_place = True
                for by in range(y, y + block_h):
                    for bx in range(x, x + block_w):
                        if self._is_in_safety_zone(bx, by):
                            safe_to_place = False
                            break
                    if not safe_to_place:
                        break
                
                if safe_to_place:
                    # Random building height between 2 and 6
                    building_h = np.random.randint(2, 7)
                    self.grid[y:y+block_h, x:x+block_w] = building_h
                    
                x += block_w + street_width
            
            y += block_h + street_width