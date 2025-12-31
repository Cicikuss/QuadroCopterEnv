import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from src.drone_env import QuadroCopterEnv
import os

# Register environment
gym.register(
    id="QuadroCopterEnv-v0",
    entry_point="src.drone_env:QuadroCopterEnv",
)

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_difficulty = 0.25  # Start with 25% difficulty (1 obstacle)
        self.difficulty_step = 0.1  # Increase by 10% each time
        self.upgrade_freq = 50_000  # Every 50k steps

    def _on_step(self) -> bool:
        # Belirli adım sayısına ulaşıldı mı?
        if self.num_timesteps % self.upgrade_freq == 0:
            # Zorluğu arttır
            self.current_difficulty += self.difficulty_step
            self.current_difficulty = min(1.0, self.current_difficulty)
            
            # Environment'a yeni zorluğu bildir
            # SB3 environment'ı wrapper içine aldığı için 'env_method' kullanıyoruz
            self.training_env.env_method("set_difficulty", self.current_difficulty)
            
            if self.verbose > 0:
                print(f"🎓 LEVEL UP! Zorluk Seviyesi: {self.current_difficulty:.1f}")
                print(f"   (Adım: {self.num_timesteps})")
                
        return True

# --- EĞİTİM ---
def train():
    print("🚀 Curriculum Learning Başlıyor...")
    print("📚 Başlangıç zorluğu: 0.25 (1 engel)")
    print("📈 Her 50k adımda +0.1 artacak")
    print("="*50)
    
    # Create environment with TimeLimit wrapper
    env = gym.make("QuadroCopterEnv-v0", size=5, render_mode=None)
    env = TimeLimit(env, max_episode_steps=300)
    
    # Set initial difficulty to 0.25 (1 obstacle out of 4)
    env.unwrapped.set_difficulty(0.25)
    
    # Create curriculum callback
    curriculum_callback = CurriculumCallback(verbose=1)
    
    # Create model
    models_dir = os.path.join(os.path.dirname(__file__), "../models/PPO")
    os.makedirs(models_dir, exist_ok=True)
    
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
    )
    
    # Train with curriculum
    model.learn(total_timesteps=500_000, callback=curriculum_callback)
    
    # Save final model
    model.save(os.path.join(models_dir, "drone_pilot_curriculum"))
    print("\n✅ Eğitim Tamamlandı!")
    print(f"📁 Model kaydedildi: {models_dir}/drone_pilot_curriculum.zip")

if __name__ == "__main__":
    train()