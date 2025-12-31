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
    COLLISION_REWARD = -50.0  # Reduced from -100 to make learning easier
    MAX_FUEL = 1000  # Increased to allow more exploration
    TARGET_REACHED_REWARD = 100.0
    DISTANCE_REWARD_SCALE = 5.0  # Increased to make distance progress more important
    STEP_PENALTY = -0.01  # Reduced to focus more on distance
    FUEL_PENALTY = -50.0  # Reduced from -100
    
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
        self.difficulty = 0.0  # Start with NO obstacles for easier learning
        self.prev_distance = 0.0
        self.last_reward = 0.0
        
        # Action smoothing parameters for continuous action space refinement
        self.smooth_action = np.array([0, 0], dtype=np.float32)
        self.alpha = 0.5  # Smoothing factor (0.5 = equal blend of old and new)
        
        # Initialize renderer if needed
        self.renderer = None
        if render_mode in ["human", "rgb_array"]:
            self.renderer = Renderer(self.window_size, self.scale_factor, self.metadata)
        self.trail = []
        self._setup_spaces()
    
    def _generate_obstacles(self) -> list:
        """Generate random obstacles that don't overlap with agent or target."""
        obstacles = []
        max_attempts = 100
        current_num_obstacles = int(self.NUM_OBSTACLES * self.difficulty)
        if current_num_obstacles == 0:
            return []
            
        for _ in range(current_num_obstacles):
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
    
    def set_difficulty(self, level: float) -> None:
        """Set difficulty level which affects number of obstacles."""
        self.difficulty = np.clip(level, 0.0, 1.0)
        print(f"🔧 Difficulty set to {self.difficulty:.2f}")
    
    def _point_in_obstacle(self, point: np.ndarray, obstacle: np.ndarray) -> bool:
        """Check if a point is inside an obstacle (with margin)."""
        x, y, w, h = obstacle
        margin = self.AGENT_HALF_SIZE * 1.5  # Slightly larger margin for safety
        
        return (x - margin < point[0] < x + w + margin and
                y - margin < point[1] < y + h + margin)
    
    def _find_safe_position(self, avoid_positions: list = None) -> np.ndarray:
        """Find a safe position that doesn't collide with obstacles or other positions."""
        max_attempts = 1000
        margin = self.AGENT_HALF_SIZE * 3  # Increased margin
        
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
                    # Ensure minimum distance between agent and target
                    min_distance = 1.5  # At least 1.5 units apart
                    if np.linalg.norm(position - other_pos) < min_distance:
                        is_safe = False
                        break
            
            if is_safe:
                return position
        
        # Fallback to center if no safe position found
        return np.array([self.size / 2, self.size / 2], dtype=np.float32)
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        # Add distance to target in observation for easier learning
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
            "relative_target": gym.spaces.Box(
                low=-self.size, high=self.size, shape=(2,), dtype=np.float32
            ),
            "distance_to_target": gym.spaces.Box(
                low=0, high=self.size * np.sqrt(2), shape=(1,), dtype=np.float32
            ),
        })
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
    
    def _get_obstacle_corners(self) -> list:
        """Convert rectangular obstacles (x, y, w, h) to corner points for LIDAR."""
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
        """Get current observation with additional helpful features."""
        obstacle_corners = self._get_obstacle_corners()
        lidar_readings = self.lidar.cast_rays(
            self._agent_location, 
            0.0,
            obstacle_corners
        )
        normalized_readings = np.array(lidar_readings) / self.lidar.max_range
        
        # Add relative position to target (helps learning direction)
        relative_target = self._target_location - self._agent_location
        distance_to_target = np.linalg.norm(relative_target)
        
        return {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
            "lidar": normalized_readings,
            "fuel": np.array([self.fuel / self.MAX_FUEL], dtype=np.float32),
            "relative_target": relative_target,
            "distance_to_target": np.array([distance_to_target], dtype=np.float32),
        }

    def _get_info(self) -> Dict[str, float]:
        """Get current info."""
        distance = np.linalg.norm(self._agent_location - self._target_location)
        return {
            "distance": distance,
            "fuel_remaining": self.fuel,
            "success": distance < self.TARGET_DISTANCE_THRESHOLD
        }
    
    def _is_path_possible(self) -> bool:
        """Check if there's a valid path between agent and target using BFS."""
        grid_resolution = 20
        grid = np.zeros((grid_resolution, grid_resolution))
        
        # Mark obstacles on grid
        for obs in self.obstacles:
            x, y, w, h = obs
            ix = int(x / self.size * grid_resolution)
            iy = int(y / self.size * grid_resolution)
            iw = int(w / self.size * grid_resolution) + 1
            ih = int(h / self.size * grid_resolution) + 1
            
            ix = np.clip(ix, 0, grid_resolution - 1)
            iy = np.clip(iy, 0, grid_resolution - 1)
            iw = np.clip(iw, 0, grid_resolution - ix)
            ih = np.clip(ih, 0, grid_resolution - iy)
            
            grid[ix:ix+iw, iy:iy+ih] = 1

        # Convert positions to grid coordinates
        start_node = (
            np.clip(int(self._agent_location[0] / self.size * grid_resolution), 0, grid_resolution - 1),
            np.clip(int(self._agent_location[1] / self.size * grid_resolution), 0, grid_resolution - 1)
        )
        target_node = (
            np.clip(int(self._target_location[0] / self.size * grid_resolution), 0, grid_resolution - 1),
            np.clip(int(self._target_location[1] / self.size * grid_resolution), 0, grid_resolution - 1)
        )

        # Check if start or target is on an obstacle
        if grid[start_node] == 1 or grid[target_node] == 1:
            return False

        # BFS pathfinding
        queue = [start_node]
        visited = set()
        visited.add(start_node)
        
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]  # Added diagonals

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
        
        # Try up to 20 times to generate a valid map
        for attempt in range(20):
            # Generate obstacles
            self.obstacles = self._generate_obstacles()
            
            # Find safe positions for agent and target
            self._agent_location = self._find_safe_position()
            self._target_location = self._find_safe_position(
                avoid_positions=[self._agent_location]
            )
            
            # Check if path exists between agent and target
            if self.difficulty == 0 or self._is_path_possible():
                break
        
        self.fuel = self.MAX_FUEL
        self.trail = []
        self.prev_distance = np.linalg.norm(self._agent_location - self._target_location)
        self.smooth_action = np.array([0, 0], dtype=np.float32)  # Reset action smoothing
        
        return self._get_obs(), self._get_info()
    
    def _check_collision(self, proposed_position: np.ndarray) -> bool:
        """Check if agent collides with obstacles at proposed position."""
        # Use a smaller collision box for more forgiving collisions
        agent_size = self.AGENT_HALF_SIZE * 0.8  # 20% smaller hitbox
        agent_l = proposed_position[0] - agent_size
        agent_r = proposed_position[0] + agent_size
        agent_t = proposed_position[1] - agent_size
        agent_b = proposed_position[1] + agent_size
        
        # Check collision with obstacles
        for obstacle in self.obstacles:
            x, y, w, h = obstacle
            if (agent_r > x and agent_l < x + w and
                agent_b > y and agent_t < y + h):
                return True
        
        return False
    
    def _calculate_reward(
        self, terminated: bool, hit_wall: bool, prev_distance: float, curr_distance: float, no_movement: bool
    ) -> float:
        """Calculate reward with shaped rewards for better learning."""
        
        # Terminal rewards/penalties
        if hit_wall:
            # Give partial reward for getting close before collision
            closeness_bonus = max(0, (5.0 - prev_distance) * 2.0)
            return self.COLLISION_REWARD + closeness_bonus
        
        if self.fuel <= 0:
            # Penalty for running out of fuel
            closeness_bonus = max(0, (5.0 - prev_distance) * 2.0)
            return self.FUEL_PENALTY + closeness_bonus
        
        if terminated:  # Target reached
            # Bonus for remaining fuel (encourages efficiency)
            fuel_bonus = (self.fuel / self.MAX_FUEL) * 20.0
            return self.TARGET_REACHED_REWARD + fuel_bonus
        
        # Distance-based shaping
        distance_improvement = prev_distance - curr_distance
        distance_reward = distance_improvement * self.DISTANCE_REWARD_SCALE
        
        # Penalty for not making progress (staying in place)
        if no_movement:
            distance_reward -= 2.0  # Strong penalty for staying still
        
        # Bonus for getting very close to target
        if curr_distance < 1.0:
            proximity_bonus = (1.0 - curr_distance) * 2.0
            distance_reward += proximity_bonus
        
        # Penalty for being stuck close to target without reaching it
        if curr_distance < 1.0 and abs(distance_improvement) < 0.01:
            distance_reward -= 1.0  # Penalty for hovering near target without reaching it
        
        # Small penalty for each step
        total_reward = distance_reward + self.STEP_PENALTY
        
        return total_reward

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step of the environment with action smoothing."""
        self.latest_action = action.copy()
        
        # Get current distance before movement
        prev_distance = np.linalg.norm(self._agent_location - self._target_location)
        prev_position = self._agent_location.copy()
        
        # Action smoothing: blend current action with previous smoothed action
        # This creates momentum and smoother trajectories
        self.smooth_action = (self.alpha * action) + ((1 - self.alpha) * self.smooth_action)
        
        # Clamp smoothed action to [-1, 1] range
        self.smooth_action = np.clip(self.smooth_action, -1.0, 1.0)
        
        # Calculate velocity with speed scaling
        speed = 0.15  # Base speed multiplier for controlled movement
        velocity = self.smooth_action * speed
        
        # Fuel consumption based on movement magnitude (more movement = more fuel)
        movement_cost = int(np.linalg.norm(velocity) * 5) + 1  # 1-6 fuel per step
        self.fuel -= movement_cost
        
        # Calculate proposed location
        proposed_location = self._agent_location + velocity
        
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
        
        # Check if agent actually moved
        movement_magnitude = np.linalg.norm(self._agent_location - prev_position)
        no_movement = movement_magnitude < 0.001  # Agent didn't move significantly
        
        # Check if target reached
        curr_distance = np.linalg.norm(self._agent_location - self._target_location)
        target_reached = curr_distance < self.TARGET_DISTANCE_THRESHOLD
        
        terminated = target_reached or hit_wall or (self.fuel <= 0)
        truncated = False
        reward = self._calculate_reward(terminated, hit_wall, prev_distance, curr_distance, no_movement)
        self.last_reward = reward
        
        # Update trail for visualization
        if self.render_mode == "human":
            self.trail.append(self._agent_location.copy())
            if len(self.trail) > 100:
                self.trail.pop(0)
        
        self.prev_distance = curr_distance
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Render the environment using the Renderer class.
        
        Returns:
            np.ndarray: RGB array if render_mode='rgb_array', None otherwise
        """
        if self.render_mode not in ["human", "rgb_array"] or self.renderer is None:
            return None
        
        # Get LIDAR scan points
        lidar_points = []
        if hasattr(self.lidar, 'last_scan_points'):
            lidar_points = self.lidar.last_scan_points or []
        
        # Delegate rendering to Renderer
        return self.renderer.render(
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
            agent_half_size=self.AGENT_HALF_SIZE,
            current_reward=self.last_reward,
            episode_terminated=False,
            return_rgb_array=(self.render_mode == "rgb_array")
        )
    
    def close(self) -> None:
        """Close the environment and cleanup renderer."""
        if self.renderer is not None:
            self.renderer.close()