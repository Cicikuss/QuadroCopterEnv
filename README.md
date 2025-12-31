# QuadroCopterEnv 🚁

A custom Gymnasium environment for training autonomous quadcopter agents using reinforcement learning (PPO). The drone navigates through obstacles using LIDAR sensors to reach target locations.

## Features ✨

- **Custom Gymnasium Environment**: Built from scratch following OpenAI Gym standards
- **LIDAR Sensor System**: 16-ray LIDAR for obstacle detection (360° coverage)
- **Intelligent Pathfinding**: BFS-based validation ensures target is always reachable
- **Fuel Management**: Limited fuel encourages efficient path planning
- **Pygame Visualization**: Real-time rendering with LIDAR rays, fuel bar, and drone rotation
- **PPO Training**: Uses Stable-Baselines3 for reinforcement learning

## Project Structure 📁

```
QuadroCopterEnv/
├── drone_env.py          # Main Gymnasium environment
├── lidar.py              # LIDAR sensor implementation
├── train.py              # Training script (PPO)
├── test.py               # Model testing with visualization
├── images/
│   ├── quadro_copter.png # Drone sprite
│   └── target.png        # Target sprite
└── models/
    └── PPO/              # Saved models directory
```

## Installation 🔧

### Requirements

```bash
pip install gymnasium
pip install numpy
pip install pygame
pip install stable-baselines3
```

### Setup

```bash
git clone <your-repo-url>
cd QuadroCopterEnv
```

## Usage 🚀

### 1. Train the Agent

Train a new model from scratch (takes 2-5 minutes):

```bash
python train.py
```

Training parameters:
- **Total timesteps**: 100,000
- **Max episode steps**: 300
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MultiInputPolicy (handles Dict observation space)

### 2. Test the Trained Agent

Watch the trained agent navigate:

```bash
python test.py
```

This will load the trained model and visualize the drone's behavior in real-time.

### 3. Analyze Performance

Check detailed statistics:

```bash
python analyze_model.py
```

## Environment Details 🎮

### Observation Space

Dict observation with 4 components:

```python
{
    "agent": Box(0, 5, shape=(2,)),      # Drone position (x, y)
    "target": Box(0, 5, shape=(2,)),     # Target position (x, y)
    "lidar": Box(0, 3, shape=(16,)),     # 16 LIDAR ray distances (normalized)
    "fuel": Box(0, 1, shape=(1,)),       # Remaining fuel (normalized)
}
```

### Action Space

Continuous 2D movement:

```python
Box(-1, 1, shape=(2,))  # (x_velocity, y_velocity)
```

### Reward Structure

| Event | Reward | Description |
|-------|--------|-------------|
| 🎯 Target Reached | **+100** | Successfully reached the target |
| 💥 Collision | **-100** | Hit an obstacle |
| ⛽ Out of Fuel | **-100** | Ran out of fuel |
| ➡️ Moving Closer | **+2.0** | Per unit distance reduced |
| ⬅️ Moving Away | **-2.0** | Per unit distance increased |
| ⏱️ Each Step | **-0.05** | Encourages efficiency |

### Environment Constants

```python
NUM_OBSTACLES = 4                    # Number of obstacles per episode
OBSTACLE_SIZE_MIN = 0.5             # Minimum obstacle size
OBSTACLE_SIZE_MAX = 1.5             # Maximum obstacle size
TARGET_DISTANCE_THRESHOLD = 0.5     # Target reach distance
MAX_FUEL = 500                      # Starting fuel amount
AGENT_HALF_SIZE = 0.25              # Drone collision box half-size
```

## Key Features Explained 🔍

### LIDAR System

- **16 rays** cast in 360° around the drone
- **3.0 unit range** for obstacle detection
- Ray distances normalized to [0, 1] for neural network input
- Visualized as red lines during rendering

### Pathfinding Validation

- Uses **BFS (Breadth-First Search)** on a 20×20 grid
- Ensures target is always reachable at episode start
- Prevents impossible scenarios
- Retries up to 10 times to generate valid maps

### Fuel System

- Starts with **500 fuel units**
- Consumes **1 fuel per step**
- Episode terminates when fuel depletes
- Visualized as green bar in top-left corner

### Collision Detection

- **Axis-Aligned Bounding Box (AABB)** collision
- Agent size: 0.5 × 0.5 units
- Checks all obstacles every step
- Margin of 0.25 units for safe positioning

## Training Tips 💡

### Improving Performance

1. **Increase training time**: Change `TIMESTEPS` in `train.py`
   ```python
   TIMESTEPS = 500000  # More training steps
   ```

2. **Adjust reward scaling**: Modify constants in `drone_env.py`
   ```python
   DISTANCE_REWARD_SCALE = 3.0  # Stronger movement reward
   STEP_PENALTY = -0.1          # Stronger efficiency penalty
   ```

3. **Tune hyperparameters**: Modify PPO parameters
   ```python
   model = PPO("MultiInputPolicy", env, 
               learning_rate=0.0003,
               n_steps=2048,
               batch_size=64,
               verbose=1)
   ```

## Troubleshooting 🔧

### Common Issues

**1. Agent doesn't move**
- Retrain the model with `python train.py`
- Check that `render_mode=None` (not `"none"`) in training

**2. Gymnasium import error**
- Install: `pip install gymnasium` (not `gym`)

**3. LIDAR rays not visible**
- Check that images are in `images/` folder
- LIDAR only shows in `render_mode="human"`

**4. Model not found**
- Run `python train.py` first to create the model
- Check `models/PPO/drone_pilot_final.zip` exists

## Performance Metrics 📊

Expected results after 100k training steps:

- **Success Rate**: 40-60%
- **Average Steps to Target**: 80-120 steps
- **Collision Rate**: 10-20%
- **Fuel Depletion**: <5%

## Future Enhancements 🚀

Potential improvements:

- [ ] Dynamic obstacle movement
- [ ] Multi-target waypoint navigation
- [ ] Continuous action space refinement
- [ ] Diagonal LIDAR rays
- [ ] Wind/drift physics simulation
- [ ] Curriculum learning (easy → hard maps)
- [ ] 3D environment extension

## Contributing 🤝

Feel free to:
- Report bugs via Issues
- Suggest features
- Submit pull requests
- Share your trained models

## License 📄

MIT License - Feel free to use this project for learning and development.

## Acknowledgments 🙏

- **Gymnasium**: OpenAI's RL environment framework
- **Stable-Baselines3**: High-quality RL algorithm implementations
- **Pygame**: Visualization library

---

**Happy Training!** 🎮🤖

For questions or issues, please open an issue on GitHub.
