"""Training script for robotic manipulation tasks using RL algorithms (current: SAC)."""

import os
import re
import argparse
from pathlib import Path
from datetime import datetime
import yaml

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import HerReplayBuffer, DDPG, TD3, SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from reward_wrapper import ShapedRewardWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)


class RewardComponentCallback(BaseCallback):
    """Logs reward components to TensorBoard at each episode end."""

    def _on_step(self):
        for info in self.locals["infos"]:
            if "ep_dense" in info:
                self.logger.record("reward/ep_dense", info["ep_dense"])
                self.logger.record("reward/ep_grasp_quality_mean", info["ep_grasp_quality_mean"])
        return True


def load_config():
    yaml_filename = f"{CONFIG['model_class']}_{CONFIG['env_id']}.yaml"
    config_path = os.path.join("hyperparams", yaml_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Hyperparameters config file not found: {config_path}")

    print(f"Reading hyperparameters from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "replay_buffer_class" in config:
        config["replay_buffer_class"] = HerReplayBuffer
    return config


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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint zip to resume from (e.g. logs/.../checkpoints/rl_model_200000_steps.zip)",
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

CONFIG.update(load_config())

env_name = CONFIG["env_id"]
base_dir = os.path.join(CONFIG["log_dir"], env_name)

if args.resume:
    # reuse existing run dir so logs are continuous
    # checkpoint path: logs/<env>/<run>/checkpoints/rl_model_<step>_steps.zip
    run_dir = str(Path(args.resume).parent.parent)
    match = re.search(r"rl_model_(\d+)_steps", args.resume)
    completed_steps = int(match.group(1)) if match else 0
    remaining_steps = CONFIG["total_timesteps"] - completed_steps
    print(f"Resuming from step {completed_steps:,}, {remaining_steps:,} steps remaining.")
else:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{CONFIG['model_class']}_{timestamp}")
    completed_steps = 0
    remaining_steps = CONFIG["total_timesteps"]

CONFIG.update(
    {
        "checkpoint_dir": os.path.join(run_dir, "checkpoints"),
        "tensorboard_log_dir": os.path.join(base_dir, "tensorboard"),
    }
)

for dir_path in [CONFIG["checkpoint_dir"], CONFIG["tensorboard_log_dir"]]:
    os.makedirs(dir_path, exist_ok=True)

# environment setup
gym.register_envs(gymnasium_robotics)
env = ShapedRewardWrapper(gym.make(CONFIG["env_id"]))
eval_env = Monitor(ShapedRewardWrapper(gym.make(CONFIG["env_id"])))
env.reset(seed=CONFIG["seed"])
env.action_space.seed(CONFIG["seed"])

# save replay buffer alongside each checkpoint so resume is fast
checkpoint_callback = CheckpointCallback(
    save_freq=CONFIG["checkpoint_freq"],
    save_path=CONFIG["checkpoint_dir"],
    save_replay_buffer=True,
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=run_dir,
    log_path=run_dir,
    eval_freq=CONFIG["eval_freq"],
)
callback = CallbackList([checkpoint_callback, eval_callback, RewardComponentCallback()])

model_class = {
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}[CONFIG["model_class"]]

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=CONFIG["action_noise_sigma"] * np.ones(n_actions)
)

if args.resume:
    model = model_class.load(
        args.resume,
        env=env,
        verbose=CONFIG["verbose"],
        tensorboard_log=CONFIG["tensorboard_log_dir"],
    )
    model.action_noise = action_noise
    buffer_path = args.resume.replace(".zip", "_replay_buffer.pkl")
    if os.path.exists(buffer_path):
        model.load_replay_buffer(buffer_path)
        print(f"Replay buffer loaded: {model.replay_buffer.size():,} transitions")
    else:
        print("No replay buffer found — starting with empty buffer.")
else:
    model = model_class(
        policy=CONFIG["policy"],
        env=env,
        buffer_size=CONFIG["buffer_size"],
        batch_size=CONFIG["batch_size"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"],
        learning_rate=CONFIG["learning_rate"],
        learning_starts=CONFIG.get("learning_starts", 100),
        replay_buffer_class=CONFIG.get("replay_buffer_class", DictReplayBuffer),
        replay_buffer_kwargs=CONFIG.get("replay_buffer_kwargs"),
        verbose=CONFIG["verbose"],
        action_noise=action_noise,
        tensorboard_log=CONFIG["tensorboard_log_dir"],
        policy_kwargs=CONFIG["policy_kwargs"],
        seed=CONFIG["seed"],
    )

try:
    model.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        reset_num_timesteps=not bool(args.resume),
    )
    print("\nTraining completed. Saving model...")

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving model...")

finally:
    model_path = os.path.join(run_dir, env_name)
    model.save(model_path)
    model.save_replay_buffer(f"{model_path}_buffer")
    print(f"Model and replay buffer saved to: {run_dir}")
    env.close()
    eval_env.close()
