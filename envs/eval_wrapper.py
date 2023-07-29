from robomimic.envs.env_base import EnvType
from utils.env_util import *
from collections import deque

class OmnimimicEvalWrapper:
    def __init__(self, env, obs_modalities, num_frames=1):
        self.env = env
        self.obs_modalities = obs_modalities
        if num_frames > 1:
            self.stacked_obs = True
            self.num_frames = num_frames
            self.obs_history = None
        else:
            self.stacked_obs = False
        self.reset()

    def step(self, action):
        """
        Take a step in the environment with an input action, return (observation, reward, done, info).
        """
        raw_obs, reward, done, info = self.env.step(action)
        obs_processed = process_omni_obs(raw_obs, self.obs_modalities)
        if self.stacked_obs:
            for mod in self.obs_modalities:
                self.obs_history[mod].append(obs_processed[mod])
            self.obs_history.append(obs_processed)
            obs = self._stack_observation()
        else:
            obs = obs_processed
        return obs, reward, done, info

    def reset(self):
        """
        Reset the environment, return observation
        """                                                                                                                                                                           
        raw_obs = self.env.reset()
        obs_processed = process_omni_obs(raw_obs, self.obs_modalities)
        if self.stacked_obs:
            self.obs_history = {}
            for mod in self.obs_modalities:
                self.obs_history[mod] = deque(
                    [obs_processed[mod] for _ in range(self.num_frames)], 
                    maxlen=self.num_frames,
                )
            obs = self._stack_observation()
        else:
            obs = obs_processed
        return obs

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render the environment if mode=='human'. Return an RGB array if mode=='rgb_array'
        """
        pass

    def get_observation(self, obs=None):
        """
        Return the current environment observation as a dictionary, unless obs is not None. 
        This function should process the raw environment observation to align with the input expected by the policy model. 
        For example, it should cast an image observation to float with value range 0-1 and shape format [C, H, W].
        """
        raw_obs = self.env.observation_spec()
        obs = process_omni_obs(raw_obs, self.obs_modalities)
        return obs

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary { str: bool } 
        with at least a “task” key for the overall task success, and additional optional keys corresponding to other task criteria.
        """
        raise NotImplementedError

    def _stack_observation(self):
        """
        Stacks the current observation with the previous n-1 observations, where n = self.num_frames.
        """
        return {mod: np.stack(self.obs_history[mod], axis=0) for mod in self.obs_modalities}
