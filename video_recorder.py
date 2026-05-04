# modified from https://gymnasium.farama.org/introduction/record_agent/

import os
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

# configuration
env_id = "FetchPickAndPlace-v4"
algo = "SAC"
num_eval_episodes = 3
log_dir = "./logs/" + env_id


# find and load the trained model
def get_latest_model_path(log_dir, env_name):
    """Find the latest model based on timestamp."""
    algo_folders = [d for d in os.listdir(log_dir) if d.startswith(f"{algo}_")]
    if not algo_folders:
        raise FileNotFoundError(f"No {algo} folders found in {log_dir}")

    latest_algo_folder = max(algo_folders)
    run_dir = os.path.join(log_dir, latest_algo_folder)
    # prefer final model, fall back to best_model saved by EvalCallback
    final = os.path.join(run_dir, env_name)
    best = os.path.join(run_dir, "best_model")
    return final if os.path.exists(final + ".zip") else best


# setup environment with video recording
env = gym.make(env_id, render_mode="rgb_array")
env = RecordVideo(
    env, video_folder="videos", name_prefix="eval", episode_trigger=lambda x: True
)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

# load the trained model
model_path = get_latest_model_path(log_dir, env_id)
print(f"Loading model from: {model_path}")
model = SAC.load(model_path, env=env)

# evaluation loop
for episode_num in range(num_eval_episodes):
    obs, info = env.reset()
    episode_reward = 0
    episode_over = False

    while not episode_over:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_over = terminated or truncated

    print(
        f"Episode {episode_num + 1}: Reward = {episode_reward:.2f}, "
        f"Success = {info.get('is_success', 0)}"
    )

env.close()

print("\nEvaluation Results:")
print(f"Episode time taken: {env.time_queue}")
print(f"Episode total rewards: {env.return_queue}")
print(f"Episode lengths: {env.length_queue}")
