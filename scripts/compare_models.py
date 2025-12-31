import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from gymnasium.wrappers import TimeLimit
from src.drone_env import QuadroCopterEnv
import numpy as np

# Register environment
gym.register(
    id="QuadroCopterEnv-v0",
    entry_point="src.drone_env:QuadroCopterEnv",
)

def evaluate_model(model, env, num_episodes=50, model_name="Model"):
    """Evaluate a model over multiple episodes."""
    print(f"\n🔍 {model_name} Değerlendiriliyor... ({num_episodes} episode)")
    
    successes = 0
    collisions = 0
    timeouts = 0
    fuel_outs = 0
    total_steps = []
    total_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                distance = info.get("distance", 1.0)
                
                if distance < 0.5:
                    successes += 1
                elif env.unwrapped.fuel <= 0:
                    fuel_outs += 1
                else:
                    collisions += 1
                
                total_steps.append(steps)
                total_rewards.append(episode_reward)
                break
            
            if truncated:
                timeouts += 1
                break
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{num_episodes} tamamlandı...")
    
    # Calculate statistics
    success_rate = (successes / num_episodes) * 100
    collision_rate = (collisions / num_episodes) * 100
    timeout_rate = (timeouts / num_episodes) * 100
    fuel_out_rate = (fuel_outs / num_episodes) * 100
    avg_steps = np.mean(total_steps)
    avg_reward = np.mean(total_rewards)
    std_steps = np.std(total_steps)
    std_reward = np.std(total_rewards)
    
    return {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "fuel_out_rate": fuel_out_rate,
        "avg_steps": avg_steps,
        "std_steps": std_steps,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "successes": successes,
    }

def compare_models():
    print("=" * 70)
    print("🏆 PPO vs SAC Model Karşılaştırması")
    print("=" * 70)
    
    # Create evaluation environment
    env = gym.make("QuadroCopterEnv-v0", size=5, render_mode=None)
    env = TimeLimit(env, max_episode_steps=300)
    
    # Test at different difficulty levels
    difficulty_levels = [
        (0.25, "Kolay (1 engel)"),
        (0.5, "Orta (2 engel)"),
        (0.75, "Zor (3 engel)"),
        (1.0, "Çok Zor (4 engel)")
    ]
    
    # Load models
    try:
        ppo_model = PPO.load("models/PPO/drone_pilot_curriculum", env=env)
        print("✅ PPO modeli yüklendi")
    except FileNotFoundError:
        print("❌ PPO modeli bulunamadı!")
        ppo_model = None
    
    try:
        sac_model = SAC.load("models/SAC/drone_pilot_sac", env=env)
        print("✅ SAC modeli yüklendi")
    except FileNotFoundError:
        print("❌ SAC modeli bulunamadı!")
        sac_model = None
    
    if not ppo_model and not sac_model:
        print("\n⚠️ Hiçbir model bulunamadı!")
        return
    
    # Evaluate at each difficulty
    for difficulty, diff_name in difficulty_levels:
        print("\n" + "="*70)
        print(f"📊 Zorluk Seviyesi: {diff_name}")
        print("="*70)
        
        env.unwrapped.set_difficulty(difficulty)
        
        results = {}
        
        if ppo_model:
            results["PPO"] = evaluate_model(ppo_model, env, num_episodes=50, model_name="PPO")
        
        if sac_model:
            results["SAC"] = evaluate_model(sac_model, env, num_episodes=50, model_name="SAC")
        
        # Print comparison
        print("\n📈 Sonuçlar:")
        print("-" * 70)
        print(f"{'Metrik':<25} {'PPO':>20} {'SAC':>20}")
        print("-" * 70)
        
        if "PPO" in results and "SAC" in results:
            ppo = results["PPO"]
            sac = results["SAC"]
            
            print(f"{'Başarı Oranı (%)':<25} {ppo['success_rate']:>19.1f}% {sac['success_rate']:>19.1f}%")
            print(f"{'Çarpışma Oranı (%)':<25} {ppo['collision_rate']:>19.1f}% {sac['collision_rate']:>19.1f}%")
            print(f"{'Ortalama Adım':<25} {ppo['avg_steps']:>19.1f} {sac['avg_steps']:>19.1f}")
            print(f"{'Std. Sapma (Adım)':<25} {ppo['std_steps']:>19.1f} {sac['std_steps']:>19.1f}")
            print(f"{'Ortalama Reward':<25} {ppo['avg_reward']:>19.1f} {sac['avg_reward']:>19.1f}")
            print(f"{'Std. Sapma (Reward)':<25} {ppo['std_reward']:>19.1f} {sac['std_reward']:>19.1f}")
            
            # Determine winner
            print("\n🏆 Kazanan:")
            if sac['success_rate'] > ppo['success_rate']:
                print(f"  SAC daha başarılı! ({sac['success_rate']:.1f}% vs {ppo['success_rate']:.1f}%)")
            elif ppo['success_rate'] > sac['success_rate']:
                print(f"  PPO daha başarılı! ({ppo['success_rate']:.1f}% vs {sac['success_rate']:.1f}%)")
            else:
                print(f"  Berabere! ({ppo['success_rate']:.1f}%)")
        
        elif "PPO" in results:
            ppo = results["PPO"]
            print(f"{'Başarı Oranı (%)':<25} {ppo['success_rate']:>19.1f}%")
            print(f"{'Ortalama Adım':<25} {ppo['avg_steps']:>19.1f}")
            print(f"{'Ortalama Reward':<25} {ppo['avg_reward']:>19.1f}")
        
        elif "SAC" in results:
            sac = results["SAC"]
            print(f"{'Başarı Oranı (%)':<25} {sac['success_rate']:>19.1f}%")
            print(f"{'Ortalama Adım':<25} {sac['avg_steps']:>19.1f}")
            print(f"{'Ortalama Reward':<25} {sac['avg_reward']:>19.1f}")
    
    env.close()
    print("\n" + "="*70)
    print("✅ Karşılaştırma Tamamlandı!")
    print("="*70)

if __name__ == "__main__":
    compare_models()
