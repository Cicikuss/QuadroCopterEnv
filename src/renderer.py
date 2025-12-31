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
        agent_half_size: float = 0.25
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
        
        # Draw LIDAR rays
        self._draw_lidar_rays(canvas, agent_pos, lidar_points)
        
        # Draw shadow
        self._draw_shadow(canvas, agent_pos)
        
        # Draw trail
        if trail:
            self._draw_trail(canvas, trail)
        
        # Draw agent (drone)
        rotation = self._get_rotation(latest_action)
        self._draw_agent(canvas, agent_pos, rotation)
        
        # Draw target
        self._draw_target(canvas, target_pos)
        
        # Draw debug hitbox if enabled
        if debug_mode:
            self._draw_hitbox(canvas, agent_pos, agent_half_size)
        
        # Draw fuel bar and text
        self._draw_fuel_bar(canvas, fuel, max_fuel)
        self._draw_fuel_text(canvas, fuel, max_fuel)
        
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
        """Draw LIDAR rays with endpoint circles."""
        if not lidar_points:
            return
        
        for point in lidar_points:
            end_pos = (
                int(point[0] * self.scale_factor), 
                int(point[1] * self.scale_factor)
            )
            # Draw ray line
            pygame.draw.line(canvas, (255, 0, 0), agent_pos, end_pos, 1)
            # Draw endpoint dot
            pygame.draw.circle(canvas, (255, 0, 0), end_pos, 3)
    
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
    
    def close(self) -> None:
        """Close the renderer and cleanup."""
        if self._initialized and self.window is not None:
            pygame.quit()
            self._initialized = False
