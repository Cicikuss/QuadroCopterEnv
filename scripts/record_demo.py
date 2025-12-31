"""
Utility script to record a demo GIF of the trained agent.
Requires: pip install imageio imageio-ffmpeg pillow
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from src.drone_env import QuadroCopterEnv
import numpy as np
import imageio
import pygame

# Register environment
gym.register(
    id="QuadroCopterEnv-v0",
    entry_point="src.drone_env:QuadroCopterEnv",
)

def record_demo(model_type="PPO", difficulty=1.0, num_episodes=3, output_path="images/demo.gif"):
    """Record a demo GIF of the trained agent.
    
    Args:
        model_type: "PPO" or "SAC"
        difficulty: 0.25, 0.5, 0.75, or 1.0
        num_episodes: Number of episodes to record
        output_path: Path to save the GIF
    """
    print(f"🎬 {model_type} Demo kaydediliyor...")
    print(f"   Zorluk: {difficulty*100:.0f}% ({int(4*difficulty)} engel)")
    print(f"   Episode sayısı: {num_episodes}")
    
    # Create environment with rgb_array mode for recording
    env = gym.make("QuadroCopterEnv-v0", size=5, render_mode="rgb_array")
    env.unwrapped.set_difficulty(difficulty)
    
    # Load model
    try:
        if model_type == "PPO":
            model = PPO.load("models/PPO/drone_pilot_curriculum", env=env)
        elif model_type == "SAC":
            model = SAC.load("models/SAC/drone_pilot_sac", env=env)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        print(f"✅ {model_type} model yüklendi")
    except FileNotFoundError:
        print(f"❌ {model_type} modeli bulunamadı!")
        return
    
    # Record episodes
    frames = []
    episode_count = 0
    
    obs, info = env.reset()
    
    print("\n📹 Kayıt başladı...")
    
    while episode_count < num_episodes:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and capture frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Check if episode ended
        if terminated or truncated:
            episode_count += 1
            print(f"   Episode {episode_count}/{num_episodes} kaydedildi")
            
            if episode_count < num_episodes:
                obs, info = env.reset()
    
    env.close()
    
    if not frames:
        print("❌ Hiç frame kaydedilemedi!")
        print("   Muhtemelen render() None döndürüyor.")
        return
    
    # Save as GIF
    print(f"\n💾 GIF oluşturuluyor: {output_path}")
    
    # Reduce frame rate and optimize
    # Take every 2nd frame to reduce size
    optimized_frames = frames[::2]
    
    # Save with imageio
    imageio.mimsave(
        output_path,
        optimized_frames,
        fps=15,  # 15 FPS for smooth but small file
        loop=0   # Infinite loop
    )
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"✅ GIF kaydedildi!")
    print(f"   Toplam frame: {len(frames)} -> {len(optimized_frames)} (optimized)")
    print(f"   Dosya boyutu: {file_size:.2f} MB")
    print(f"   Konum: {output_path}")

def main():
    print("="*70)
    print("🎬 Demo GIF Oluşturucu")
    print("="*70)
    
    # Check if models exist
    ppo_exists = Path("models/PPO/drone_pilot_curriculum.zip").exists()
    sac_exists = Path("models/SAC/drone_pilot_sac.zip").exists()
    
    if not ppo_exists and not sac_exists:
        print("\n❌ Hiçbir model bulunamadı!")
        print("   Önce train.py veya train_sac.py ile model eğitin.")
        return
    
    # Model selection
    print("\n🤖 Hangi modeli kaydetmek istersiniz?")
    if ppo_exists:
        print("  1 - PPO")
    if sac_exists:
        print("  2 - SAC")
    
    choice = input("Seçim (1 veya 2): ").strip()
    
    if choice == "1" and ppo_exists:
        model_type = "PPO"
    elif choice == "2" and sac_exists:
        model_type = "SAC"
    else:
        print("❌ Geçersiz seçim!")
        return
    
    # Difficulty selection
    print("\n🎮 Zorluk Seviyesi:")
    print("  1 - Kolay (1 engel)")
    print("  2 - Orta (2 engel)")
    print("  3 - Zor (3 engel)")
    print("  4 - Çok Zor (4 engel)")
    
    diff_choice = input("Seçim (1-4) [Enter=3]: ").strip() or "3"
    difficulty_map = {"1": 0.25, "2": 0.5, "3": 0.75, "4": 1.0}
    difficulty = difficulty_map.get(diff_choice, 0.75)
    
    # Number of episodes
    num_episodes = input("\nKaç episode kaydetmek istersiniz? [Enter=3]: ").strip()
    num_episodes = int(num_episodes) if num_episodes else 3
    
    # Record
    print()
    record_demo(model_type=model_type, difficulty=difficulty, num_episodes=num_episodes)
    
    print("\n💡 GIF'i README'de görmek için:")
    print("   1. Git commit yapın")
    print("   2. GitHub'a push edin")
    print("   3. README.md'deki demo bölümü otomatik görünecek")

if __name__ == "__main__":
    try:
        import imageio
    except ImportError:
        print("❌ imageio kütüphanesi bulunamadı!")
        print("   Yüklemek için: pip install imageio imageio-ffmpeg pillow")
        sys.exit(1)
    
    main()
