import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import SAC
from src.drone_env import QuadroCopterEnv

# Register environment
gym.register(
    id="QuadroCopterEnv-v0",
    entry_point="src.drone_env:QuadroCopterEnv",
)

def test():
    print("👀 SAC Model yükleniyor ve test başlıyor...")
    
    # Difficulty selection
    print("\n🎮 Zorluk Seviyesi Seçin:")
    print("  1 - Kolay (1 engel)")
    print("  2 - Orta (2 engel)")
    print("  3 - Zor (3 engel)")
    print("  4 - Çok Zor (4 engel - Maximum)")
    
    difficulty_choice = input("Seçim (1-4) [Enter=4]: ").strip() or "4"
    difficulty_map = {"1": 0.25, "2": 0.5, "3": 0.75, "4": 1.0}
    difficulty = difficulty_map.get(difficulty_choice, 1.0)
    
    print(f"\n✅ Zorluk ayarlandı: {difficulty*100:.0f}% ({int(4*difficulty)} engel)")

    # Create environment
    env = gym.make("QuadroCopterEnv-v0", size=5, render_mode="human")
    env.unwrapped.set_difficulty(difficulty)

    # Load trained SAC model
    model_path = "models/SAC/drone_pilot_sac"
    try:
        model = SAC.load(model_path, env=env)
        print(f"✅ SAC model yüklendi: {model_path}.zip")
    except FileNotFoundError:
        print("❌ Model dosyası bulunamadı! Önce train_sac.py'yi çalıştır.")
        print("   Aranılan: models/SAC/drone_pilot_sac.zip")
        return

    # Simulation loop
    obs, info = env.reset()
    episode_count = 0
    total_reward = 0.0
    step_count = 0
    
    print("\n🎮 Test başladı... (ESC veya pencereyi kapat = çıkış)")
    
    for i in range(10000):  # Watch for many episodes
        # deterministic=True: Agent uses best learned action (no exploration)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Render manually (Gymnasium standard: separate from step())
        env.render()

        if terminated:
            episode_count += 1
            avg_reward = total_reward / step_count if step_count > 0 else 0
            
            if info.get("distance", 1.0) < 0.5:
                print(f"🎉 Episode {episode_count}: Hedefe Ulaşıldı! (Adım: {step_count}, Ort. Reward: {avg_reward:.2f})")
            else:
                print(f"💥 Episode {episode_count}: Başarısız (Adım: {step_count}, Ort. Reward: {avg_reward:.2f})")
            
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0
        
        elif truncated:  # TimeLimit reached
            episode_count += 1
            print(f"⏱️ Episode {episode_count}: Zaman Aşımı (Adım: {step_count})")
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0

    print("Test bitti.")
    env.close()

if __name__ == "__main__":
    test()
