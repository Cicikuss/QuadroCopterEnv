import gymnasium as gym
from stable_baselines3 import PPO
from drone_env import QuadroCopterEnv
import numpy as np

# Register environment
gym.register(
    id="QuadroCopterEnv-v0",
    entry_point="drone_env:QuadroCopterEnv",
)

def analyze_model():
    print("🔍 Model Analizi")
    print("=" * 60)
    
    # Load environment
    env = gym.make("QuadroCopterEnv-v0", size=5, render_mode=None)
    
    # Try to load model
    model_path = "models/PPO/drone_pilot_final"
    try:
        model = PPO.load(model_path, env=env)
        print(f"✅ Model yüklendi: {model_path}.zip")
        print(f"Policy: {model.policy}")
        print(f"Total timesteps trained: {model.num_timesteps}")
    except FileNotFoundError:
        print(f"❌ Model bulunamadı: {model_path}.zip")
        print("Eğitim yapılmamış olabilir. train.py çalıştır.")
        env.close()
        return
    
    print("\n📊 Model Test (Deterministic Mode)")
    print("=" * 60)
    
    obs, info = env.reset()
    
    total_reward = 0
    distances = []
    actions_taken = []
    
    for i in range(100):
        # Get model prediction
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        distances.append(info['distance'])
        actions_taken.append(action)
        
        if i % 10 == 0:
            print(f"Step {i:3d} | Action: [{action[0]:6.2f}, {action[1]:6.2f}] | "
                  f"Reward: {reward:7.2f} | Distance: {info['distance']:6.2f}")
        
        if terminated:
            print(f"\n✅ Episode ended at step {i}")
            break
    
    print("\n" + "=" * 60)
    print("📈 İstatistikler:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Avg Reward per step: {total_reward / len(distances):.3f}")
    print(f"Avg Distance: {np.mean(distances):.3f}")
    print(f"Min Distance: {np.min(distances):.3f}")
    print(f"Max Distance: {np.max(distances):.3f}")
    
    # Analyze actions
    actions_array = np.array(actions_taken)
    print(f"\nAction Analysis:")
    print(f"Avg Action X: {np.mean(actions_array[:, 0]):.3f}")
    print(f"Avg Action Y: {np.mean(actions_array[:, 1]):.3f}")
    print(f"Action magnitude: {np.mean(np.linalg.norm(actions_array, axis=1)):.3f}")
    
    # Check if agent is moving at all
    initial_distance = distances[0]
    final_distance = distances[-1]
    
    if abs(initial_distance - final_distance) < 0.1:
        print(f"\n⚠️  UYARI: Agent hareket etmiyor (Distance change: {final_distance - initial_distance:.3f})")
    else:
        print(f"\n✅ Agent hareket ediyor (Distance change: {final_distance - initial_distance:.3f})")
    
    env.close()

if __name__ == "__main__":
    analyze_model()
