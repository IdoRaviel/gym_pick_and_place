"""Training script for robotic manipulation tasks using RL algorithms (current: SAC)."""

import os
import argparse
from datetime import datetime
import yaml

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import HerReplayBuffer, DDPG, TD3, SAC
from reward_wrapper import ShapedRewardWrapper
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)


# load configuration from YAML file
def load_config():
    yaml_filename = f"{CONFIG['model_class']}_{CONFIG['env_id']}.yaml"
    config_path = os.path.join("hyperparams", yaml_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Hyperparameters config file not found: {config_path}")

    print(f"Reading hyperparameters from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # convert replay_buffer_class from string to actual class
    if "replay_buffer_class" in config:
        config["replay_buffer_class"] = HerReplayBuffer
    return config


# argument parser for flexibility
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL agent for FetchPush environment."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SAC",
        choices=["DDPG", "TD3", "SAC"],
        help="RL model to use (DDPG, TD3, SAC)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="FetchPickAndPlace-v4",
        help="Gymnasium environment ID",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Base directory for logs"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0: no output, 1: info, 2: debug)",
    )
    return parser.parse_args()


args = parse_args()

CONFIG = {
    "model_class": args.model,
    "env_id": args.env,
    "seed": args.seed,
    "log_dir": args.log_dir,
    "verbose": args.verbose,
}

# update CONFIG with the loaded config
CONFIG.update(load_config())

# organizing logs
env_name = CONFIG["env_id"]
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

# create base directories
base_dir = os.path.join(CONFIG["log_dir"], env_name)
run_dir = os.path.join(base_dir, f"{CONFIG['model_class']}_{timestamp}")

# set up directory structure
CONFIG.update(
    {
        "checkpoint_dir": os.path.join(run_dir, "checkpoints"),
        "tensorboard_log_dir": os.path.join(base_dir, "tensorboard"),
    }
)

# setup logging directories
for dir_path in [CONFIG["checkpoint_dir"], CONFIG["tensorboard_log_dir"]]:
    os.makedirs(dir_path, exist_ok=True)

# environment setup
gym.register_envs(gymnasium_robotics)
env = ShapedRewardWrapper(gym.make(CONFIG["env_id"]))
env.reset(seed=CONFIG["seed"])
env.action_space.seed(
    CONFIG["seed"]
)  # Seed action space for more deterministic behavior

# callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=CONFIG["checkpoint_freq"], save_path=CONFIG["checkpoint_dir"]
)
eval_callback = EvalCallback(
    env,
    best_model_save_path=run_dir,
    log_path=run_dir,
    eval_freq=CONFIG["eval_freq"],
)
callback = CallbackList([checkpoint_callback, eval_callback])

# model setup
model_class = {
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}[CONFIG["model_class"]]

# action noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=CONFIG["action_noise_sigma"] * np.ones(n_actions)
)

# model initialization
model = model_class(
    policy=CONFIG["policy"],
    env=env,
    buffer_size=CONFIG["buffer_size"],
    batch_size=CONFIG["batch_size"],
    gamma=CONFIG["gamma"],
    tau=CONFIG["tau"],
    learning_rate=CONFIG["learning_rate"],
    learning_starts=CONFIG.get("learning_starts", 100),
    replay_buffer_class=CONFIG.get("replay_buffer_class"),
    replay_buffer_kwargs=CONFIG.get("replay_buffer_kwargs"),
    verbose=CONFIG["verbose"],
    action_noise=action_noise,
    tensorboard_log=CONFIG["tensorboard_log_dir"],
    policy_kwargs=CONFIG["policy_kwargs"],
    seed=CONFIG["seed"],
)

# training loop
try:
    model.learn(total_timesteps=CONFIG["total_timesteps"], callback=callback)
    print("\nTraining completed. Saving model...")

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving model...")

finally:
    # saving model and replay buffer
    model_name = f"{env_name}"
    model_path = os.path.join(run_dir, model_name)
    model.save(model_path)
    model.save_replay_buffer(f"{model_path}_buffer")
    print(f"Model and replay buffer saved to: {run_dir}")

    # cleanup
    env.close()
