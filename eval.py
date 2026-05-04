"""Evaluation script for trained RL models on robotic manipulation tasks. (currently using SAC)"""

import os
import argparse
import time

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import numpy as np
from stable_baselines3 import DDPG, TD3, SAC

# configuration
CONFIG = {
    "env_id": "FetchPickAndPlace-v4",
    "model_class": "SAC",
    "seed": 1,
    "n_eval_episodes": 10,
    "render": True,
    "log_dir": "./logs",
}


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent.")
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["model_class"],
        choices=["DDPG", "TD3", "SAC"],
        help="RL model type",
    )
    parser.add_argument(
        "--env", type=str, default=CONFIG["env_id"], help="Gymnasium environment ID"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=CONFIG["n_eval_episodes"],
        help="Number of evaluation episodes",
    )
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--record", action="store_true", help="Record video to ./videos/ instead of live render")
    return parser.parse_args()


# parse arguments and update config
args = parse_args()
CONFIG.update(
    {
        "model_class": args.model,
        "env_id": args.env,
        "seed": args.seed,
        "n_eval_episodes": args.episodes,
        "render": not args.no_render,
        "record": args.record,
    }
)

# setup model paths
env_name = CONFIG["env_id"]
model_name = CONFIG["model_class"]
CONFIG["log_dir"] = os.path.join(CONFIG["log_dir"], env_name)


# find latest folder and model path
def get_latest_model_path(log_dir, env_name):
    """Find the latest SAC model based on timestamp."""
    model_folders = [d for d in os.listdir(log_dir) if d.startswith(f"{model_name}_")]
    if not model_folders:
        raise FileNotFoundError(f"No {model_name} folders found in {log_dir}")

    latest_sac_folder = max(
        model_folders
    )  # Latest timestamp will be last alphabetically
    model_path = os.path.join(log_dir, latest_sac_folder, env_name)
    return model_path


model_path = get_latest_model_path(CONFIG["log_dir"], env_name)


# environment setup
gym.register_envs(gymnasium_robotics)
if CONFIG["record"]:
    render_mode = "rgb_array"
elif CONFIG["render"]:
    render_mode = "human"
else:
    render_mode = None
env = gym.make(CONFIG["env_id"], render_mode=render_mode)
if CONFIG["record"]:
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda _: True)
env.reset(seed=CONFIG["seed"])

# load model
model_class = {
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}[CONFIG["model_class"]]

print(f"Loading model from: {model_path}")
model = model_class.load(model_path, env=env)

# evaluation loop
rewards = []
successes = []

try:
    for episode in range(CONFIG["n_eval_episodes"]):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            time.sleep(1 / 25)

        rewards.append(episode_reward)
        successes.append(info.get("is_success", 0))

        print(
            f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
            f"Success = {info.get('is_success', 0)}"
        )

except KeyboardInterrupt:
    print("\nEvaluation interrupted by user.")

finally:
    # print summary statistics
    if rewards:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        success_rate = np.mean(successes) * 100

        print("\nEvaluation Results:")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Success rate: {success_rate:.1f}%")

    # cleanup
    env.close()
