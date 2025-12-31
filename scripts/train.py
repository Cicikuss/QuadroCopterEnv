import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from src.drone_env import QuadroCopterEnv
import os

# 1. Ortamı Kayıt Et
gym.register(
    id="QuadroCopterEnv-v0",
    entry_point="src.drone_env:QuadroCopterEnv",
)

def train():
    # Klasör temizliği (Eski model varsa kafası karışmasın)
    models_dir = "models/PPO"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("🚀 Eğitim Başlıyor... (Görsellik Kapalı - Hızlı Mod)")

    # 2. Ortamı Yarat (render_mode=None -> HIZ İÇİN ÖNEMLİ)
    env = gym.make("QuadroCopterEnv-v0", size=5, render_mode="human", debug_mode=True  )
    
    # 3. Zaman Sınırı Ekle (arttırıldı: engelleri dolaşması için daha fazla zamanı var)
    # 300 adım = target'a ulaşmak için yeterli, ama sonsuz loop'u engeller
    env = TimeLimit(env, max_episode_steps=300)

    # 4. Modeli Seç
    # MultiInputPolicy: Çünkü observation space'imiz bir Dict (Agent + Target)
    model = PPO("MultiInputPolicy", env, verbose=1)

    # 5. Eğit (Örn: 100.000 adım)
    # Bilgisayarının hızına göre 1-3 dakika sürebilir.
    TIMESTEPS = 100000
    model.learn(total_timesteps=TIMESTEPS)

    # 6. Kaydet
    model_path = f"{models_dir}/drone_pilot_final"
    model.save(model_path)
    print(f"✅ Eğitim bitti! Model şuraya kaydedildi: {model_path}.zip")
    
    env.close()

if __name__ == "__main__":
    train()