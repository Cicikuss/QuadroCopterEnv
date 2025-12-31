import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from src.drone_env import QuadroCopterEnv

# 1. Ortamı Kayıt Et
gym.register(
    id="QuadroCopterEnv-v0",
    entry_point="src.drone_env:QuadroCopterEnv",
)

def test():
    print("👀 Model yükleniyor ve test başlıyor...")

    # 2. Ortamı Yarat (render_mode="human" -> İZLEMEK İÇİN)
    env = gym.make("QuadroCopterEnv-v0", size=5, render_mode="human")

    # 3. Eğitilmiş Modeli Yükle
    model_path = "models/PPO/drone_pilot_final"
    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print("❌ Model dosyası bulunamadı! Önce train.py'yi çalıştır.")
        return

    # 4. Simülasyon Döngüsü
    obs, info = env.reset()
    
    for i in range(1000): # 1000 adım boyunca izleyelim
        # deterministic=True: Ajan öğrendiği EN İYİ hamleyi yapar (macera aramaz)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render manually (Gymnasium standard: separate from step())
        env.render()

        if terminated:
            print(f"🎉 Hedefe Ulaşıldı! (Adım: {i})")
            obs, info = env.reset()
        
        elif truncated: # Eğer TimeLimit kullanırsan burası çalışır
            print("timeout - reset")
            obs, info = env.reset()

    print("Test bitti.")
    env.close()

if __name__ == "__main__":
    test()