import numpy as np
import gymnasium as gym


FINGER_GEOMS = {"robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link"}
OBJECT_GEOM = "object0"


class ShapedRewardWrapper(gym.Wrapper):
    """
    Custom reward shaping for FetchPickAndPlace:
      (1) Dense distance penalty  — gripper→object + object→goal every step
      (2) Sparse grasp bonus      — one-time when MuJoCo contacts confirm grasp
      (3) Sparse lift bonus       — one-time when grasped object clears the table
    """

    GRASP_BONUS = 1.0
    LIFT_REWARD_PER_STEP = 0.1    # continuous reward each step while grasping+lifted
    LIFT_HEIGHT_THRESHOLD = 0.05  # meters above starting object height

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._grasp_bonus_given = False
        self._initial_obj_z = obs["observation"][5]
        # build geom-id → geom-name map once per episode (model doesn't change)
        self._geom_names = self._build_geom_name_map()
        self._ep_dense = 0.0
        self._ep_grasp_bonus = 0.0
        self._ep_lift = 0.0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        dense, grasp_bonus, lift_reward = self._reward_components(obs)
        reward = dense + grasp_bonus + lift_reward

        self._ep_dense += dense
        self._ep_grasp_bonus += grasp_bonus
        self._ep_lift += lift_reward

        # write episode totals to info so callbacks can log them to TensorBoard
        if terminated or truncated:
            info["ep_dense"] = self._ep_dense
            info["ep_grasp_bonus"] = self._ep_grasp_bonus
            info["ep_lift"] = self._ep_lift
            info["grasp_triggered"] = float(self._grasp_bonus_given)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _build_geom_name_map(self):
        model = self.env.unwrapped.model
        return {i: model.geom(i).name for i in range(model.ngeom)}

    def _is_grasping(self):
        """True if either finger geom is currently in contact with the object geom."""
        data = self.env.unwrapped.data
        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = self._geom_names.get(contact.geom1, "")
            g2 = self._geom_names.get(contact.geom2, "")
            pair = {g1, g2}
            if OBJECT_GEOM in pair and pair & FINGER_GEOMS:
                return True
        return False

    # ------------------------------------------------------------------
    def _reward_components(self, obs):
        o = obs["observation"]
        achieved = obs["achieved_goal"]  # object (x,y,z)
        desired = obs["desired_goal"]    # target (x,y,z)

        # observation layout (FetchPickAndPlace-v4):
        # [0:3]  gripper position
        # [3:6]  object position
        # [6:9]  object position relative to gripper
        object_pos = o[3:6]
        object_rel_pos = o[6:9]

        # (1) dense: pull gripper toward object, then object toward goal
        dist_grip_obj = np.linalg.norm(object_rel_pos)
        dist_obj_goal = np.linalg.norm(achieved - desired)
        dense = -(dist_grip_obj + dist_obj_goal)

        # (2) sparse grasp bonus — real MuJoCo contact check, one-time per episode
        grasping = self._is_grasping()
        grasp_bonus = 0.0
        if grasping and not self._grasp_bonus_given:
            grasp_bonus = self.GRASP_BONUS
            self._grasp_bonus_given = True

        # (3) continuous lift reward — earned every step while grasping AND lifted
        # drops to 0 immediately if cube is dropped or falls below threshold
        lifted = object_pos[2] > self._initial_obj_z + self.LIFT_HEIGHT_THRESHOLD
        lift_reward = self.LIFT_REWARD_PER_STEP if (grasping and lifted) else 0.0

        return dense, grasp_bonus, lift_reward
