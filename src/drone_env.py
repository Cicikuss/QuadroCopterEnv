import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any
from src.lidar import Lidar
from src.renderer import Renderer


class QuadroCopterEnv(gym.Env):
    # Constants
    NUM_OBSTACLES = 4
    OBSTACLE_SIZE_MIN = 0.5
    OBSTACLE_SIZE_MAX = 1.5
    WINDOW_SIZE = 800
    AGENT_HALF_SIZE = 0.15
    TARGET_DISTANCE_THRESHOLD = 0.5
    COLLISION_REWARD = -100.0
    MAX_FUEL = 500  # Increased from 1000, but enough for 300 steps with efficiency bonus
    TARGET_REACHED_REWARD = 100.0
    DISTANCE_REWARD_SCALE = 2.0  # Stronger reward for moving closer (0-2.0 per step)
    STEP_PENALTY = -0.05  # Small penalty per step to encourage efficiency
    FUEL_PENALTY = -0.1  # Penalty for running out of fuel
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, size: int = 5, render_mode: Optional[str] = None, debug_mode: bool = False) -> None:
        self.size = size
        self.window_size = self.WINDOW_SIZE
        self.scale_factor = self.window_size / self.size
        self.render_mode = render_mode
        self.debug_mode = debug_mode
        self.fuel = self.MAX_FUEL
        self.lidar = Lidar(max_range=3.0, num_rays=16, fov_angle=360)
        self.obstacles = []
        self.latest_action = np.array([0, 0], dtype=np.float32)
        self._agent_location = np.array([-1, -1], dtype=np.float32)
        self._target_location = np.array([-1, -1], dtype=np.float32)
        
        # Initialize renderer only if needed
        self.renderer = None
        if render_mode == "human":
            self.renderer = Renderer(self.window_size, self.scale_factor, self.metadata)
        self.trail = []
        self._setup_spaces()
    
    def _generate_obstacles(self) -> list:
        """Generate random obstacles that don't overlap with agent or target."""
        obstacles = []
        max_attempts = 100
        
        for _ in range(self.NUM_OBSTACLES):
            attempts = 0
            while attempts < max_attempts:
                x = self.np_random.uniform(0, self.size - self.OBSTACLE_SIZE_MAX)
                y = self.np_random.uniform(0, self.size - self.OBSTACLE_SIZE_MAX)
                w = self.np_random.uniform(self.OBSTACLE_SIZE_MIN, self.OBSTACLE_SIZE_MAX)
                h = self.np_random.uniform(self.OBSTACLE_SIZE_MIN, self.OBSTACLE_SIZE_MAX)
                
                # Ensure obstacle doesn't exceed boundaries
                if x + w > self.size:
                    w = self.size - x
                if y + h > self.size:
                    h = self.size - y
                
                obstacle = np.array([x, y, w, h], dtype=np.float32)
                
                # Check if obstacle overlaps with agent or target (only if they're initialized)
                agent_valid = (self._agent_location[0] >= 0)
                target_valid = (self._target_location[0] >= 0)
                
                overlaps_agent = agent_valid and self._point_in_obstacle(self._agent_location, obstacle)
                overlaps_target = target_valid and self._point_in_obstacle(self._target_location, obstacle)
                
                if not overlaps_agent and not overlaps_target:
                    obstacles.append(obstacle)
                    break
                
                attempts += 1
        
        return obstacles
    
    def _point_in_obstacle(self, point: np.ndarray, obstacle: np.ndarray) -> bool:
        """Check if a point is inside an obstacle (with margin)."""
        x, y, w, h = obstacle
        margin = self.AGENT_HALF_SIZE
        
        return (x - margin < point[0] < x + w + margin and
                y - margin < point[1] < y + h + margin)
    
    def _find_safe_position(self, avoid_positions: list = None) -> np.ndarray:
        """Find a safe position that doesn't collide with obstacles or other positions."""
        max_attempts = 1000
        margin = self.AGENT_HALF_SIZE * 2
        
        for _ in range(max_attempts):
            # Generate random position with margin from boundaries
            position = self.np_random.uniform(
                margin, 
                self.size - margin, 
                size=2
            ).astype(np.float32)
            
            # Check if position is safe from obstacles
            is_safe = True
            for obstacle in self.obstacles:
                if self._point_in_obstacle(position, obstacle):
                    is_safe = False
                    break
            
            # Check if position is far enough from other positions
            if is_safe and avoid_positions:
                for other_pos in avoid_positions:
                    if np.linalg.norm(position - other_pos) < margin * 2:
                        is_safe = False
                        break
            
            if is_safe:
                return position
        
        # Fallback to center if no safe position found
        return np.array([self.size / 2, self.size / 2], dtype=np.float32)
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=0, high=self.size, shape=(2,), dtype=np.float32
            ),
            "target": gym.spaces.Box(
                low=0, high=self.size, shape=(2,), dtype=np.float32
            ),
            "lidar": gym.spaces.Box(
                low=0, high=self.lidar.max_range, shape=(self.lidar.num_rays,), dtype=np.float32
            ),
            "fuel": gym.spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            ),
        })
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
    
    def _get_obstacle_corners(self) -> list:
        """Convert rectangular obstacles (x, y, w, h) to corner points for LIDAR.
        
        Returns:
            List of corner points for each obstacle: [[(x1,y1), (x2,y2), (x3,y3), (x4,y4)], ...]
        """
        corners_list = []
        for obs in self.obstacles:
            x, y, w, h = obs
            corners = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h)
            ]
            corners_list.append(corners)
        return corners_list
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        obstacle_corners = self._get_obstacle_corners()
        lidar_readings = self.lidar.cast_rays(
            self._agent_location, 
            0.0,
            obstacle_corners
        )
        normalized_readings = np.array(lidar_readings) / self.lidar.max_range
        
        return {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
            "lidar": normalized_readings,
            "fuel": np.array([self.fuel / self.MAX_FUEL], dtype=np.float32)
        }

    def _get_info(self) -> Dict[str, float]:
        """Get current info."""
        distance = np.linalg.norm(self._agent_location - self._target_location)
        return {"distance": distance}
    def _is_path_possible(self) -> bool:
        """Check if there's a valid path between agent and target using BFS (Flood Fill).
        
        Returns:
            True if path exists, False otherwise
        """
        grid_resolution = 20
        grid = np.zeros((grid_resolution, grid_resolution))
        
        # Mark obstacles on grid
        for obs in self.obstacles:
            x, y, w, h = obs
            ix = int(x / self.size * grid_resolution)
            iy = int(y / self.size * grid_resolution)
            iw = int(w / self.size * grid_resolution) + 1
            ih = int(h / self.size * grid_resolution) + 1
            
            grid[ix:ix+iw, iy:iy+ih] = 1

        # Convert positions to grid coordinates
        start_node = (
            int(self._agent_location[0] / self.size * grid_resolution),
            int(self._agent_location[1] / self.size * grid_resolution)
        )
        target_node = (
            int(self._target_location[0] / self.size * grid_resolution),
            int(self._target_location[1] / self.size * grid_resolution)
        )

        # Check if start or target is on an obstacle
        try:
            if grid[start_node] == 1 or grid[target_node] == 1:
                return False
        except IndexError:
            return False

        # BFS pathfinding
        queue = [start_node]
        visited = set()
        visited.add(start_node)
        
        directions = [(0,1), (0,-1), (1,0), (-1,0)]

        while queue:
            current = queue.pop(0)
            
            if current == target_node:
                return True
            
            for dx, dy in directions:
                next_node = (current[0] + dx, current[1] + dy)
                
                if (0 <= next_node[0] < grid_resolution and 
                    0 <= next_node[1] < grid_resolution and 
                    next_node not in visited and 
                    grid[next_node] == 0):
                    
                    visited.add(next_node)
                    queue.append(next_node)
                    
        return False

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Try up to 10 times to generate a valid map with reachable target
        for _ in range(10):
            # Generate obstacles
            self.obstacles = self._generate_obstacles()
            
            # Find safe positions for agent and target
            self._agent_location = self._find_safe_position()
            self._target_location = self._find_safe_position(
                avoid_positions=[self._agent_location]
            )
            
            # Check if path exists between agent and target
            if self._is_path_possible():
                break
        
        self.fuel = self.MAX_FUEL
        return self._get_obs(), self._get_info()
    
    def _check_collision(self, proposed_position: np.ndarray) -> bool:
        """Check if agent collides with obstacles at proposed position."""
        # Agent dimensions
        agent_w, agent_h = 0.5, 0.5
        agent_l = proposed_position[0] - agent_w / 2
        agent_r = proposed_position[0] + agent_w / 2
        agent_t = proposed_position[1] - agent_h / 2
        agent_b = proposed_position[1] + agent_h / 2
        
        # Check collision with obstacles
        for obstacle in self.obstacles:
            x, y, w, h = obstacle
            if (agent_r > x and agent_l < x + w and
                agent_b > y and agent_t < y + h):
                return True
        
        return False
    
    def _calculate_reward(
        self, terminated: bool, hit_wall: bool, prev_distance: float, curr_distance: float
    ) -> float:
        """Calculate reward based on environment state.
        
        Reward structure:
        - Target reached (without collision): +100
        - Collision: -100
        - Out of fuel: -100
        - Moving closer: distance_diff (positive reward for reducing distance)
        - Moving away: -distance_diff (penalty for increasing distance)
        - Each step: -0.05 (encourage efficiency)
        """
        # Highest priority: collision and fuel are worst
        if hit_wall:
            return self.COLLISION_REWARD  # -100
        
        if self.fuel <= 0:
            return self.FUEL_PENALTY  # -100
        
        # Second priority: reaching target is best
        if terminated:  # This means target reached (not from collision/fuel)
            return self.TARGET_REACHED_REWARD  # +100
        
        # Otherwise: reward movement toward target
        # If moved closer: positive reward, if moved away: negative penalty
        distance_diff = prev_distance - curr_distance  # positive if moved closer
        distance_reward = distance_diff * self.DISTANCE_REWARD_SCALE  # Can be +/- 2.0
        
        # Add small penalty for each step to encourage quick solutions
        step_penalty = self.STEP_PENALTY  # -0.05
        
        return distance_reward + step_penalty

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step of the environment."""
        self.latest_action = action.copy()
        self.trail.append(self._agent_location.copy())
        if len(self.trail) > 10: # keep last 10 positions
            self.trail.pop(0)
        # Get current distance before movement
        prev_distance = np.linalg.norm(self._agent_location - self._target_location)
        
        # Speed multiplier for controlled movement (increased from 0.2 to 0.5)
        speed = 0.5
        self.fuel -= 1  # Decrease fuel per step
        # Calculate proposed location
        proposed_location = self._agent_location + (action * speed)
        
        # Clip to boundaries with margin
        margin = self.AGENT_HALF_SIZE
        clipped_location = np.clip(
            proposed_location,
            margin,
            self.size - margin
        )
        
        # Check collision at clipped location
        hit_wall = self._check_collision(clipped_location)
        
        # Update position if no collision
        if not hit_wall:
            self._agent_location = clipped_location
        
        # Check if target reached
        curr_distance = np.linalg.norm(self._agent_location - self._target_location)
        target_reached = curr_distance < self.TARGET_DISTANCE_THRESHOLD
        
        terminated = target_reached or hit_wall or (self.fuel <= 0)
        truncated = False
        reward = self._calculate_reward(terminated, hit_wall, prev_distance, curr_distance)
        
        # Update trail for visualization
        if self.render_mode == "human":
            self.trail.append(self._agent_location.copy())
            if len(self.trail) > 100:  # Keep last 100 positions
                self.trail.pop(0)
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> None:
        """Render the environment using the Renderer class."""
        if self.render_mode != "human" or self.renderer is None:
            return
        
        # Get LIDAR scan points
        lidar_points = []
        if hasattr(self.lidar, 'last_scan_points'):
            lidar_points = self.lidar.last_scan_points or []
        
        # Delegate rendering to Renderer
        self.renderer.render(
            agent_location=self._agent_location,
            target_location=self._target_location,
            obstacles=self.obstacles,
            lidar_points=lidar_points,
            latest_action=self.latest_action,
            fuel=self.fuel,
            max_fuel=self.MAX_FUEL,
            size=self.size,
            trail=self.trail,
            debug_mode=self.debug_mode,
            agent_half_size=self.AGENT_HALF_SIZE
        )
    
    def close(self) -> None:
        """Close the environment and cleanup renderer."""
        if self.renderer is not None:
            self.renderer.close()