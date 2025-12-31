import numpy as np
import pygame
from typing import Tuple, List, Any


class Renderer:
    """Handles all rendering operations for the QuadroCopter environment."""
    
    def __init__(self, window_size: int, scale_factor: float, metadata: dict):
        self.window_size = window_size
        self.scale_factor = scale_factor
        self.metadata = metadata
        
        self.window = None
        self.clock = None
        self.quadro_copter_image_original = None
        self.target_image = None
        
        self._initialized = False
        
        # Statistics tracking
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_steps = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.fps_samples = []
    
    def initialize(self) -> None:
        """Initialize pygame window and clock."""
        if self._initialized:
            return
        
        pygame.init()
        pygame.display.set_caption("QuadroCopterEnv")
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        
        self._initialized = True
    
    def load_images(self) -> None:
        """Load and scale game images."""
        if self.quadro_copter_image_original is None:
            self.quadro_copter_image_original = pygame.transform.scale(
                pygame.image.load("images/quadro_copter.png").convert_alpha(),
                (
                    int(0.5 * self.scale_factor),
                    int(0.5 * self.scale_factor),
                ),
            )
        
        if self.target_image is None:
            self.target_image = pygame.transform.scale(
                pygame.image.load("images/target.png").convert_alpha(),
                (
                    int(0.5 * self.scale_factor),
                    int(0.5 * self.scale_factor),
                ),
            )
    
    def render(
        self,
        agent_location: np.ndarray,
        target_location: np.ndarray,
        obstacles: List[np.ndarray],
        lidar_points: List[Tuple[float, float]],
        latest_action: np.ndarray,
        fuel: int,
        max_fuel: int,
        size: float,
        trail: List[np.ndarray] = None,
        debug_mode: bool = False,
        agent_half_size: float = 0.25,
        current_reward: float = 0.0,
        episode_terminated: bool = False
    ) -> None:
        """Render the environment.
        
        Args:
            agent_location: Current agent position (x, y)
            target_location: Target position (x, y)
            obstacles: List of obstacles [(x, y, w, h), ...]
            lidar_points: List of LIDAR scan endpoint positions
            latest_action: Latest action taken (for rotation)
            fuel: Current fuel amount
            max_fuel: Maximum fuel amount
            size: Environment size
            trail: List of previous agent positions for trail visualization
            debug_mode: Whether to show debug visualizations (hitboxes, etc.)
            agent_half_size: Half-size of agent collision box
        """
        if not self._initialized:
            self.initialize()
        
        self.load_images()
        
        # Create canvas
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        # Draw grid
        self._draw_grid(canvas)
        
        # Draw obstacles
        self._draw_obstacles(canvas, obstacles)
        
        # Get positions
        agent_pos = self._get_position(agent_location, size)
        target_pos = self._get_position(target_location, size)
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(agent_location - target_location)
        
        # Draw target pulse effect
        self._draw_target_pulse(canvas, target_pos)
        
        # Draw direction arrow to target
        self._draw_direction_arrow(canvas, agent_pos, target_pos)
        
        # Draw LIDAR rays
        self._draw_lidar_rays(canvas, agent_pos, lidar_points)
        
        # Draw collision warning if obstacles nearby
        self._draw_collision_warning(canvas, agent_pos, lidar_points)
        
        # Draw shadow
        self._draw_shadow(canvas, agent_pos)
        
        # Draw trail
        if trail:
            self._draw_trail(canvas, trail)
        
        # Draw agent (drone)
        rotation = self._get_rotation(latest_action)
        self._draw_agent(canvas, agent_pos, rotation)
        
        # Draw velocity arrow
        self._draw_velocity_arrow(canvas, agent_pos, latest_action)
        
        # Draw target
        self._draw_target(canvas, target_pos)
        
        # Draw debug hitbox if enabled
        if debug_mode:
            self._draw_hitbox(canvas, agent_pos, agent_half_size)
        
        # Draw HUD elements
        self._draw_fuel_bar(canvas, fuel, max_fuel)
        self._draw_fuel_text(canvas, fuel, max_fuel)
        self._draw_distance_indicator(canvas, distance_to_target)
        self._draw_episode_stats(canvas, current_reward)
        
        # Update statistics
        self.current_episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += current_reward
        
        if episode_terminated:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_steps = 0
            self.current_episode_reward = 0.0
        
        # Update display
        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
    
    def _draw_obstacles(self, canvas: pygame.Surface, obstacles: List[np.ndarray]) -> None:
        """Draw obstacles on canvas."""
        for obstacle in obstacles:
            rectx = int(obstacle[0] * self.scale_factor)
            recty = int(obstacle[1] * self.scale_factor)
            rectw = int(obstacle[2] * self.scale_factor)
            recth = int(obstacle[3] * self.scale_factor)
            pygame.draw.rect(
                canvas, (128, 128, 128), pygame.Rect(rectx, recty, rectw, recth)
            )
    
    def _draw_grid(self, canvas: pygame.Surface) -> None:
        """Draw background grid."""
        grid_size = int(self.scale_factor)
        for x in range(0, self.window_size, grid_size):
            pygame.draw.line(canvas, (240, 240, 240), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, grid_size):
            pygame.draw.line(canvas, (240, 240, 240), (0, y), (self.window_size, y))
    
    def _get_position(self, location: np.ndarray, size: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        x = np.clip(
            location[0] * self.scale_factor,
            0,
            (size - 0.01) * self.scale_factor,
        )
        y = np.clip(
            location[1] * self.scale_factor,
            0,
            (size - 0.01) * self.scale_factor,
        )
        return (int(x), int(y))
    
    def _draw_lidar_rays(
        self, 
        canvas: pygame.Surface, 
        agent_pos: Tuple[int, int], 
        lidar_points: List[Tuple[float, float]]
    ) -> None:
        """Draw LIDAR rays with distance-based color gradient and scanning area.
        
        - Scanning area: Semi-transparent circle showing max LIDAR range
        - Ray colors: Green (far) -> Yellow (medium) -> Red (close)
        - Hit points: Larger circles at obstacle intersections
        - Ray thickness: Proportional to distance (thicker = closer obstacle)
        """
        if not lidar_points:
            return
        
        # Draw LIDAR max range circle (semi-transparent)
        max_range_pixels = int(3.0 * self.scale_factor)  # 3.0 is max_range from Lidar
        range_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        pygame.draw.circle(
            range_surface, 
            (100, 150, 255, 30),  # Light blue, very transparent
            agent_pos, 
            max_range_pixels, 
            1
        )
        canvas.blit(range_surface, (0, 0))
        
        # Draw each ray with color gradient based on distance
        for point in lidar_points:
            end_pos = (
                int(point[0] * self.scale_factor), 
                int(point[1] * self.scale_factor)
            )
            
            # Calculate distance from agent to hit point
            dx = end_pos[0] - agent_pos[0]
            dy = end_pos[1] - agent_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Normalize distance (0 = close, 1 = far)
            normalized_distance = min(distance / max_range_pixels, 1.0)
            
            # Color gradient: Red (0.0) -> Yellow (0.5) -> Green (1.0)
            if normalized_distance < 0.5:
                # Red to Yellow
                t = normalized_distance * 2  # 0 to 1
                color = (255, int(255 * t), 0)
                thickness = 3
                dot_radius = 5
            else:
                # Yellow to Green
                t = (normalized_distance - 0.5) * 2  # 0 to 1
                color = (int(255 * (1 - t)), 255, int(100 * t))
                thickness = 2
                dot_radius = 4
            
            # Draw ray line with varying thickness
            pygame.draw.line(canvas, color, agent_pos, end_pos, thickness)
            
            # Draw hit point with glow effect
            # Outer glow
            glow_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*color, 50), end_pos, dot_radius + 3)
            canvas.blit(glow_surface, (0, 0))
            
            # Inner solid dot
            pygame.draw.circle(canvas, color, end_pos, dot_radius)
            pygame.draw.circle(canvas, (255, 255, 255), end_pos, dot_radius // 2)
    
    def _get_rotation(self, action: np.ndarray) -> float:
        """Calculate rotation angle based on action."""
        if np.linalg.norm(action) > 0:
            angle = np.arctan2(action[1], action[0])
            return -np.degrees(angle)
        return 0.0
    
    def _draw_agent(self, canvas: pygame.Surface, pos: Tuple[int, int], rotation: float) -> None:
        """Draw the agent (drone)."""
        rotated_image = pygame.transform.rotate(
            self.quadro_copter_image_original, rotation
        )
        canvas.blit(
            rotated_image,
            (
                pos[0] - rotated_image.get_width() // 2,
                pos[1] - rotated_image.get_height() // 2,
            ),
        )
    
    def _draw_target(self, canvas: pygame.Surface, pos: Tuple[int, int]) -> None:
        """Draw the target."""
        canvas.blit(
            self.target_image,
            (
                pos[0] - self.target_image.get_width() // 2,
                pos[1] - self.target_image.get_height() // 2,
            ),
        )
    
    def _draw_fuel_bar(self, canvas: pygame.Surface, fuel: int, max_fuel: int) -> None:
        """Draw the fuel bar."""
        fuel_bar_width = int(200 * (fuel / max_fuel))
        pygame.draw.rect(canvas, (0, 255, 0), pygame.Rect(10, 10, fuel_bar_width, 20))
        pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(10, 10, 200, 20), 2)
    
    def _draw_fuel_text(self, canvas: pygame.Surface, fuel: int, max_fuel: int) -> None:
        """Draw fuel level text."""
        font = pygame.font.SysFont(None, 24)
        fuel_text = font.render(f'Fuel: {fuel}/{max_fuel}', True, (0, 0, 0))
        canvas.blit(fuel_text, (10, 40))
    
    def _draw_shadow(self, canvas: pygame.Surface, agent_pos: Tuple[int, int]) -> None:
        """Draw shadow under agent."""
        shadow_offset = int(5 * self.scale_factor / 100)
        shadow_pos = (agent_pos[0] + shadow_offset, agent_pos[1] + shadow_offset)
        shadow_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surface, (0, 0, 0, 50), shadow_pos, int(0.2 * self.scale_factor))
        canvas.blit(shadow_surface, (0, 0))
    
    def _draw_trail(self, canvas: pygame.Surface, trail: List[np.ndarray]) -> None:
        """Draw agent movement trail."""
        if len(trail) > 1:
            trail_points = [
                (int(pos[0] * self.scale_factor), int(pos[1] * self.scale_factor)) 
                for pos in trail
            ]
            if len(trail_points) >= 2:
                pygame.draw.lines(canvas, (0, 0, 255), False, trail_points, 2)
    
    def _draw_hitbox(self, canvas: pygame.Surface, agent_pos: Tuple[int, int], agent_half_size: float) -> None:
        """Draw debug hitbox around agent."""
        agent_rect_pixel = pygame.Rect(
            agent_pos[0] - int(agent_half_size * self.scale_factor),
            agent_pos[1] - int(agent_half_size * self.scale_factor),
            int(agent_half_size * 2 * self.scale_factor),
            int(agent_half_size * 2 * self.scale_factor),
        )
        pygame.draw.rect(canvas, (0, 0, 255), agent_rect_pixel, 1)
    
    def _draw_direction_arrow(self, canvas: pygame.Surface, agent_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> None:
        """Draw arrow pointing from agent to target."""
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < 5:  # Too close, don't draw
            return
        
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Arrow starts at agent radius, ends 30 pixels away
        arrow_start = (int(agent_pos[0] + dx * 30), int(agent_pos[1] + dy * 30))
        arrow_end = (int(agent_pos[0] + dx * 60), int(agent_pos[1] + dy * 60))
        
        # Draw dashed line to target
        dash_length = 10
        num_dashes = int(distance / dash_length)
        for i in range(0, num_dashes, 2):
            start = (
                int(agent_pos[0] + dx * i * dash_length),
                int(agent_pos[1] + dy * i * dash_length)
            )
            end = (
                int(agent_pos[0] + dx * (i + 1) * dash_length),
                int(agent_pos[1] + dy * (i + 1) * dash_length)
            )
            pygame.draw.line(canvas, (255, 165, 0, 100), start, end, 2)
        
        # Draw arrow
        pygame.draw.line(canvas, (255, 165, 0), arrow_start, arrow_end, 3)
        
        # Arrow head
        angle = np.arctan2(dy, dx)
        arrow_head_length = 10
        left_angle = angle + np.pi * 0.75
        right_angle = angle - np.pi * 0.75
        
        left_point = (
            int(arrow_end[0] + arrow_head_length * np.cos(left_angle)),
            int(arrow_end[1] + arrow_head_length * np.sin(left_angle))
        )
        right_point = (
            int(arrow_end[0] + arrow_head_length * np.cos(right_angle)),
            int(arrow_end[1] + arrow_head_length * np.sin(right_angle))
        )
        
        pygame.draw.polygon(canvas, (255, 165, 0), [arrow_end, left_point, right_point])
    
    def _draw_velocity_arrow(self, canvas: pygame.Surface, agent_pos: Tuple[int, int], action: np.ndarray) -> None:
        """Draw velocity vector arrow."""
        magnitude = np.linalg.norm(action)
        if magnitude < 0.1:  # Too small, don't draw
            return
        
        # Scale arrow length
        arrow_length = magnitude * 50
        dx = action[0] / magnitude
        dy = action[1] / magnitude
        
        arrow_end = (
            int(agent_pos[0] + dx * arrow_length),
            int(agent_pos[1] + dy * arrow_length)
        )
        
        # Draw velocity line
        pygame.draw.line(canvas, (0, 255, 255), agent_pos, arrow_end, 2)
        
        # Arrow head
        angle = np.arctan2(dy, dx)
        arrow_head_length = 8
        left_angle = angle + np.pi * 0.75
        right_angle = angle - np.pi * 0.75
        
        left_point = (
            int(arrow_end[0] + arrow_head_length * np.cos(left_angle)),
            int(arrow_end[1] + arrow_head_length * np.sin(left_angle))
        )
        right_point = (
            int(arrow_end[0] + arrow_head_length * np.cos(right_angle)),
            int(arrow_end[1] + arrow_head_length * np.sin(right_angle))
        )
        
        pygame.draw.polygon(canvas, (0, 255, 255), [arrow_end, left_point, right_point])
    
    def _draw_collision_warning(self, canvas: pygame.Surface, agent_pos: Tuple[int, int], lidar_points: List[Tuple[float, float]]) -> None:
        """Draw warning ring if obstacles are very close."""
        if not lidar_points:
            return
        
        # Find minimum distance to any obstacle
        min_distance = float('inf')
        for point in lidar_points:
            end_pos = (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
            dx = end_pos[0] - agent_pos[0]
            dy = end_pos[1] - agent_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            min_distance = min(min_distance, distance)
        
        # Warning threshold (30 pixels)
        warning_threshold = 30
        if min_distance < warning_threshold:
            # Calculate warning intensity (0-1, 1 = very close)
            intensity = 1.0 - (min_distance / warning_threshold)
            
            # Draw pulsing warning ring
            import time
            pulse = (np.sin(time.time() * 10) + 1) / 2  # 0-1 oscillation
            alpha = int(100 * intensity * pulse)
            radius = int(40 + 20 * pulse)
            
            warning_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.circle(warning_surface, (255, 0, 0, alpha), agent_pos, radius, 3)
            canvas.blit(warning_surface, (0, 0))
    
    def _draw_target_pulse(self, canvas: pygame.Surface, target_pos: Tuple[int, int]) -> None:
        """Draw pulsing effect around target."""
        import time
        pulse = (np.sin(time.time() * 3) + 1) / 2  # 0-1 oscillation
        radius = int(30 + 15 * pulse)
        alpha = int(50 + 50 * (1 - pulse))
        
        pulse_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        pygame.draw.circle(pulse_surface, (0, 255, 0, alpha), target_pos, radius, 2)
        canvas.blit(pulse_surface, (0, 0))
    
    def _draw_distance_indicator(self, canvas: pygame.Surface, distance: float) -> None:
        """Draw distance to target indicator."""
        font = pygame.font.SysFont(None, 28)
        text = font.render(f'Distance: {distance:.2f}m', True, (50, 50, 50))
        
        # Background box
        padding = 5
        box_rect = pygame.Rect(
            self.window_size - text.get_width() - padding * 2 - 10,
            10,
            text.get_width() + padding * 2,
            text.get_height() + padding * 2
        )
        
        box_surface = pygame.Surface((box_rect.width, box_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(box_surface, (255, 255, 255, 200), box_surface.get_rect(), border_radius=5)
        pygame.draw.rect(box_surface, (100, 100, 100), box_surface.get_rect(), 2, border_radius=5)
        canvas.blit(box_surface, (box_rect.x, box_rect.y))
        
        canvas.blit(text, (box_rect.x + padding, box_rect.y + padding))
    
    def _draw_episode_stats(self, canvas: pygame.Surface, current_reward: float) -> None:
        """Draw episode statistics."""
        font = pygame.font.SysFont(None, 24)
        
        stats_lines = [
            f'Episode: {self.episode_count}',
            f'Steps: {self.current_episode_steps}',
            f'Reward: {self.current_episode_reward:.1f}',
        ]
        
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
            stats_lines.append(f'Avg (10): {avg_reward:.1f}')
        
        # Draw stats box
        y_offset = self.window_size - 120
        for i, line in enumerate(stats_lines):
            text = font.render(line, True, (50, 50, 50))
            
            # Background
            padding = 5
            box_rect = pygame.Rect(10, y_offset + i * 25, text.get_width() + padding * 2, text.get_height() + padding * 2)
            box_surface = pygame.Surface((box_rect.width, box_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(box_surface, (255, 255, 255, 200), box_surface.get_rect(), border_radius=3)
            pygame.draw.rect(box_surface, (100, 100, 100), box_surface.get_rect(), 1, border_radius=3)
            canvas.blit(box_surface, (box_rect.x, box_rect.y))
            
            canvas.blit(text, (box_rect.x + padding, box_rect.y + padding))
    
    def close(self) -> None:
        """Close the renderer and cleanup."""
        if self._initialized and self.window is not None:
            pygame.quit()
            self._initialized = False
