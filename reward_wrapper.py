import numpy as np
import gymnasium as gym

# Max Euclidean distance in the Fetch workspace:
# x,y each span ±0.15m (0.30m), z spans 0.45m → sqrt(0.3²+0.3²+0.45²) ≈ 0.62m
MAX_DIST = 0.7  # meters, rounded up for safety


class ShapedRewardWrapper(gym.Wrapper):
    """
    Shaped reward for FetchPickAndPlace. All components are negative; 0 is optimal.

    reward = -(dist_gripper→object / MAX_DIST)   ∈ [-1, 0]
           - (dist_object→goal   / MAX_DIST)     ∈ [-1, 0]

    Total reward ∈ [-2, 0], reaching 0 only when gripper is on object and object is at goal.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ep_dist_grip = 0.0
        self._ep_dist_goal = 0.0
        self._ep_steps = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        dist_grip, dist_goal = self._reward_components(obs)
        reward = -(dist_grip + dist_goal) / MAX_DIST

        self._ep_dist_grip += dist_grip
        self._ep_dist_goal += dist_goal
        self._ep_steps += 1

        n = self._ep_steps
        info["reward_dist_grip"] = -self._ep_dist_grip / (n * MAX_DIST)
        info["reward_dist_goal"] = -self._ep_dist_goal / (n * MAX_DIST)

        return obs, reward, terminated, truncated, info

    def _reward_components(self, obs):
        o = obs["observation"]
        achieved = obs["achieved_goal"]  # object (x,y,z)
        desired = obs["desired_goal"]    # target (x,y,z)

        # observation layout (FetchPickAndPlace-v4):
        # [0:3]  gripper position
        # [3:6]  object position
        # [6:9]  object position relative to gripper
        object_rel_pos = o[6:9]

        dist_grip = np.linalg.norm(object_rel_pos)   # meters
        dist_goal = np.linalg.norm(achieved - desired)  # meters

        return dist_grip, dist_goal
