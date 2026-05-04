import numpy as np
import gymnasium as gym


FINGER_GEOMS = {"robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link"}
OBJECT_GEOM = "object0"

# Max Euclidean distance in the Fetch workspace:
# x,y each span ±0.15m (0.30m), z spans 0.45m → sqrt(0.3²+0.3²+0.45²) ≈ 0.62m
MAX_DIST = 0.7  # meters, rounded up for safety


class ShapedRewardWrapper(gym.Wrapper):
    """
    Shaped reward for FetchPickAndPlace. All components are negative; 0 is optimal.

    reward = -(dist_gripper→object / MAX_DIST)   ∈ [-1, 0]
           - (dist_object→goal   / MAX_DIST)     ∈ [-1, 0]
           - (1 - grasp_quality)                 ∈ [-1, 0]

    grasp_quality: 0.0 = no contact, 0.5 = one finger, 1.0 = bilateral (both fingers)
    Total reward ∈ [-3, 0], reaching 0 only when gripper is on object,
    object is at goal, and both fingers are in contact.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._geom_names = self._build_geom_name_map()
        self._ep_dense = 0.0
        self._ep_grasp_quality = 0.0
        self._ep_steps = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        dense, grasp_quality = self._reward_components(obs)
        reward = dense - (1.0 - grasp_quality)

        self._ep_dense += dense
        self._ep_grasp_quality += grasp_quality
        self._ep_steps += 1

        if terminated or truncated:
            info["ep_dense"] = self._ep_dense
            info["ep_grasp_quality_mean"] = self._ep_grasp_quality / max(self._ep_steps, 1)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _build_geom_name_map(self):
        model = self.env.unwrapped.model
        return {i: model.geom(i).name for i in range(model.ngeom)}

    def _grasp_quality(self):
        """
        Bilateral contact (both fingers on object) = 1.0
        Unilateral contact (one finger)            = 0.5
        No contact                                 = 0.0
        """
        data = self.env.unwrapped.data
        right_contact = False
        left_contact = False
        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = self._geom_names.get(contact.geom1, "")
            g2 = self._geom_names.get(contact.geom2, "")
            pair = {g1, g2}
            if OBJECT_GEOM in pair:
                if "r_gripper_finger_link" in pair:
                    right_contact = True
                if "l_gripper_finger_link" in pair:
                    left_contact = True

        if right_contact and left_contact:
            return 1.0
        if right_contact or left_contact:
            return 0.5
        return 0.0

    def _reward_components(self, obs):
        o = obs["observation"]
        achieved = obs["achieved_goal"]  # object (x,y,z)
        desired = obs["desired_goal"]    # target (x,y,z)

        # observation layout (FetchPickAndPlace-v4):
        # [0:3]  gripper position
        # [3:6]  object position
        # [6:9]  object position relative to gripper
        object_rel_pos = o[6:9]

        dist_grip_obj = np.linalg.norm(object_rel_pos)
        dist_obj_goal = np.linalg.norm(achieved - desired)

        # normalize both distances to [0, 1] using max workspace distance
        dense = -(dist_grip_obj + dist_obj_goal) / MAX_DIST  # ∈ [-2, 0] normalised

        grasp_quality = self._grasp_quality()

        return dense, grasp_quality
