# QuadroCopterEnv 🚁

A custom Gymnasium environment for training autonomous quadcopter agents using reinforcement learning (PPO). The drone navigates through obstacles using LIDAR sensors to reach target locations.

## Features ✨

- **Custom Gymnasium Environment**: Built from scratch following OpenAI Gym standards
- **LIDAR Sensor System**: 16-ray LIDAR for obstacle detection (360° coverage)
- **Curriculum Learning**: Progressive difficulty - starts easy, gradually increases obstacles
- **Multiple RL Algorithms**: Train with PPO or SAC, compare performance
- **Action Smoothing**: Momentum-based continuous action space for realistic drone physics
- **Intelligent Pathfinding**: BFS-based validation ensures target is always reachable
- **Dynamic Fuel Management**: Fuel consumption based on movement speed
- **Advanced Visualization**: Real-time rendering with LIDAR gradient, direction arrows, collision warnings, and HUD
- **PPO Training**: Uses Stable-Baselines3 for reinforcement learning

## Project Structure 📁

```
QuadroCopterEnv/
├── src/
│   ├── __init__.py       # Package initialization
│   ├── drone_env.py      # Main Gymnasium environment
│   ├── lidar.py          # LIDAR sensor implementation
│   └── renderer.py       # Rendering pipeline
├── scripts/
│   ├── train.py          # PPO training script
│   ├── train_sac.py      # SAC training script  
│   ├── test.py           # PPO model testing with visualization
│   ├── test_sac.py       # SAC model testing with visualization
│   ├── compare_models.py # PPO vs SAC comparison
│   ├── analyze_model.py  # Model performance analysis
│   └── test_random.py    # Random action testing
├── images/
│   ├── quadro_copter.png # Drone sprite
│   └── target.png        # Target sprite
├── models/
│   └── PPO/              # Saved models directory
├── README.md             # Documentation
└── requirements.txt      # Python dependencies
```

## Installation 🔧

### Requirements

```bash
pip install -r requirements.txt
```

Or manually:

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

Train using **PPO** (recommended for beginners) or **SAC** (better for continuous control):

**PPO Training:**
```bash
python scripts/train.py
```

**SAC Training:**
```bash
python scripts/train_sac.py
```

Training parameters:
- **Total timesteps**: 500,000
- **Max episode steps**: 300
- **Algorithms**: 
  - **PPO**: On-policy, stable, good for general tasks
  - **SAC**: Off-policy, entropy-regularized, excellent for continuous control
- **Policy**: MultiInputPolicy (handles Dict observation space)
- **Curriculum**: Starts at 25% difficulty (1 obstacle), increases every 50k steps to 100% (4 obstacles)

### 2. Test the Trained Agent

Watch the trained agent navigate:

**Test PPO:**
```bash
python scripts/test.py
```

**Test SAC:**
```bash
python scripts/test_sac.py
```

Both scripts allow difficulty selection (1-4 obstacles) to test different scenarios.

### 3. Compare Models

Compare PPO vs SAC performance across different difficulty levels:

```bash
python scripts/compare_models.py
```

This evaluates both models on 50 episodes at each difficulty level and provides detailed statistics.

### 4. Analyze Performance

Check detailed statistics for a specific model:

```bash
python scripts/analyze_model.py
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

Continuous 2D movement with action smoothing:

```python
Box(-1, 1, shape=(2,))  # (x_velocity, y_velocity)
```

**Action Smoothing**: Actions are blended with previous actions using exponential moving average (alpha=0.5) to create momentum and smoother, more realistic trajectories.

### Reward Structure

| Event | Reward | Description |
|-------|--------|-------------|
| 🎯 Target Reached | **+100** | Successfully reached the target |
| 💥 Collision | **-50** | Hit an obstacle (reduced for better exploration) |
| ⛽ Out of Fuel | **-50** | Ran out of fuel |
| ➡️ Moving Closer | **+5.0** | Per unit distance reduced (increased for stronger signal) |
| ⬅️ Moving Away | **-5.0** | Per unit distance increased |
| ⏱️ Each Step | **-0.01** | Encourages efficiency |

### Environment Constants

```python
NUM_OBSTACLES = 4                    # Number of obstacles per episode
OBSTACLE_SIZE_MIN = 0.5             # Minimum obstacle size
OBSTACLE_SIZE_MAX = 1.5             # Maximum obstacle size
TARGET_DISTANCE_THRESHOLD = 0.5     # Target reach distance
MAX_FUEL = 1000                     # Starting fuel amount (increased for exploration)
AGENT_HALF_SIZE = 0.15              # Drone collision box half-size
ACTION_SMOOTHING_ALPHA = 0.5        # Momentum blending factor
```

## Key Features Explained 🔍

### PPO vs SAC 🤖

**PPO (Proximal Policy Optimization)**:
- ✅ **On-policy**: Learns from current policy only
- ✅ **Stable**: Clipped updates prevent large policy changes
- ✅ **Sample efficient**: Good for real-time learning
- ✅ **Best for**: General-purpose tasks, when stability matters
- ⚠️ Slower convergence on continuous control

**SAC (Soft Actor-Critic)**:
- ✅ **Off-policy**: Learns from replay buffer (past experiences)
- ✅ **Entropy regularization**: Encourages exploration
- ✅ **Sample efficient**: Reuses past data effectively
- ✅ **Best for**: Continuous action spaces, complex control
- ⚠️ More hyperparameters to tune

**Which to choose?**
- 🟢 **PPO**: Start here for simplicity and stability
- 🔵 **SAC**: Use for better final performance on continuous control tasks
- 🏆 **Compare**: Run `compare_models.py` to see which works better for your scenario

### Curriculum Learning 🎓

- **Progressive Training**: Agent learns with increasing difficulty
- **Difficulty Schedule**:
  - 0-50k steps: 25% difficulty (1 obstacle)
  - 50k-100k: 35% difficulty (1-2 obstacles)
  - 100k-150k: 45% difficulty (1-2 obstacles)
  - 150k-200k: 55% difficulty (2 obstacles)
  - 200k-250k: 65% difficulty (2-3 obstacles)
  - 250k-300k: 75% difficulty (3 obstacles)
  - 300k-350k: 85% difficulty (3 obstacles)
  - 350k+: 100% difficulty (4 obstacles - maximum)
- **Benefits**: Faster convergence, better final performance, more stable training

### Action Smoothing 🌊

- **Momentum-Based Control**: Actions are smoothed using exponential moving average
- **Formula**: `smooth_action = 0.5 * new_action + 0.5 * prev_smooth_action`
- **Benefits**: 
  - More realistic drone physics with inertia
  - Smoother trajectories and reduced jitter
  - Better generalization to real-world scenarios
- **Dynamic Fuel**: Fuel consumption scales with movement speed (1-6 units/step)

### Advanced Visualization 🎨

**LIDAR System**:
- **16 rays** cast in 360° around the drone
- **Distance-based color gradient**: 
  - 🔴 Red: Very close obstacles (danger)
  - 🟡 Yellow: Medium distance
  - 🟢 Green: Far obstacles (safe)
- **3.0 unit range** with semi-transparent max range circle
- **Glow effects** on hit points for better visibility

**Real-time HUD**:
- 🎯 **Direction Arrow**: Orange dashed line and arrow pointing to target
- ⚡ **Velocity Vector**: Cyan arrow showing current movement direction and speed
- 🔔 **Collision Warning**: Pulsing red ring when obstacles are dangerously close
- ✨ **Target Pulse**: Animated green pulse effect around target
- 📍 **Distance Indicator**: Real-time distance to target (top-right)
- 📊 **Episode Stats**: Episode number, steps, current reward, rolling average (bottom-left)
- ⛽ **Fuel Bar**: Green bar with numeric display (top-left)

### LIDAR System

### Pathfinding Validation

- Uses **BFS (Breadth-First Search)** on a 20×20 grid
- Ensures target is always reachable at episode start
- Prevents impossible scenarios
- Retries up to 10 times to generate valid maps

### Fuel System

- Starts with **1000 fuel units** (increased for longer exploration)
- **Dynamic consumption**: 1-6 fuel per step based on movement speed
- Episode terminates when fuel depletes
- Visualized as green bar in top-left corner
- Encourages efficient but not overly conservative movement

### Collision Detection

- **Axis-Aligned Bounding Box (AABB)** collision
- Agent size: 0.5 × 0.5 units
- Checks all obstacles every step
- Margin of 0.25 units for safe positioning

## Training Tips 💡

### Improving Performance

**1. Increase training time**: Change `TIMESTEPS` in `scripts/train.py`
   ```python
   TIMESTEPS = 500000  # More training steps
   ```

2. **Adjust reward scaling**: Modify constants in `src/drone_env.py`
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
- Retrain the model with `python scripts/train.py`
- Check that `render_mode=None` (not `"none"`) in training

**2. Gymnasium import error**
- Install: `pip install gymnasium` (not `gym`)

**3. LIDAR rays not visible**
- Check that images are in `images/` folder
- LIDAR only shows in `render_mode="human"`

**4. Model not found**
- Run `python scripts/train.py` first to create the model
- Check `models/PPO/drone_pilot_curriculum.zip` exists

**5. Import errors after restructuring**
- Make sure you're running scripts from the root directory
- Verify `src/__init__.py` exists

## Performance Metrics 📊

Expected results after 500k training steps with curriculum learning:

- **Success Rate**: 70-85% (improved with curriculum learning)
- **Average Steps to Target**: 60-100 steps
- **Collision Rate**: 5-15%
- **Fuel Depletion**: <5%

*Note: Curriculum learning significantly improves success rate compared to fixed difficulty training.*

## Future Enhancements 🚀

Potential improvements:

- [ ] Dynamic obstacle movement
- [ ] Multi-target waypoint navigation
- [ ] Diagonal LIDAR rays for denser coverage
- [ ] Wind/drift physics simulation
- [ ] 3D environment extension
- [ ] Multi-agent scenarios
- [ ] Variable obstacle shapes (circles, polygons)

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
