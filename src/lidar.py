import numpy as np

class Lidar:
    def __init__(self, max_range=3.0, num_rays=8, fov_angle=360):
        self.max_range = max_range
        self.num_rays = num_rays
        self.fov_angle = np.deg2rad(fov_angle)
        self.last_scan_points = [] # Çizim için lazım

    def cast_rays(self, agent_pos, agent_angle, obstacles_corners):
        """
        Cast rays and detect obstacles.
        
        Args:
            agent_pos: Agent position (x, y)
            agent_angle: Agent rotation angle in radians
            obstacles_corners: List of corner points for each obstacle
                Example: [ [(0,0), (1,0), (1,1), (0,1)], ... ]
        
        Returns:
            Array of ray distances to obstacles
        """
        readings = []
        self.last_scan_points = []
        
        # Calculate ray angles
        start_angle = agent_angle - (self.fov_angle / 2)
        step_angle = self.fov_angle / self.num_rays
        
        for i in range(self.num_rays):
            ray_angle = start_angle + (i * step_angle)
            
            # Calculate ray endpoint
            ray_end = (
                agent_pos[0] + self.max_range * np.cos(ray_angle),
                agent_pos[1] + self.max_range * np.sin(ray_angle),
            )

            closest_dist = self.max_range
            closest_point = ray_end
            
            # Check intersection with all obstacles
            for obs_corners in obstacles_corners:
                # Check each edge of the obstacle
                for j in range(len(obs_corners)):
                    p1 = obs_corners[j]
                    p2 = obs_corners[(j + 1) % len(obs_corners)]
                    
                    intersection = self.calculate_intersection(
                        agent_pos, ray_end, p1, p2
                    )
                    
                    if intersection:
                        # Calculate distance to intersection
                        dist = np.linalg.norm(np.array(intersection) - np.array(agent_pos))
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_point = intersection

            readings.append(closest_dist)
            self.last_scan_points.append(closest_point)
            
        return np.array(readings, dtype=np.float32)

    def calculate_intersection(self, ray_start, ray_end, obs_start, obs_end):
        x1, y1 = ray_start
        x2, y2 = ray_end
        x3, y3 = obs_start
        x4, y4 = obs_end

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None 

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        return None