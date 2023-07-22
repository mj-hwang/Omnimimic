from omnigibson.envs.env_wrapper import EnvironmentWrapper
from robomimic.envs.env_base import EnvType
from utils.env_util import *
import json
import h5py

h5py.get_config().track_order = True

class OmnimimicSkillWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for collecting data in robomimic format,
    when using action primitive skills developed in Omniverse.

    Args:
        env (OmniGibsonEnv): The environment to wrap.
    """
    def __init__(self, env, obs_modalities, path):
        self.env = env
        self.obs_modalities = obs_modalities

        self.traj_count = 0
        self.step_count = 0
        
        self.curr_obs = None
        self.current_traj_histories = []
        self.hdf5_file = h5py.File(path, 'w')
        self.hdf5_file.create_group(f"data")
        self.hdf5_file.create_group(f"mask")
        
        # TODO: update env kwargs
        self.env_args = {
            "env_name": self.env.name,
            "env_type": EnvType.GYM_TYPE,
            "env_kwargs": {},
        }
        self.hdf5_file["group"].attrs["env_args"] = json.dumps(self.env_args)

        # Run super
        super().__init__(env=env)
        self.reset()

    def execute_skill(self, skill_controller, skill_type):
        """
        Execute the skill given the skill controller, and flush data
        """
        curr_obs_processed = process_observation(self.curr_obs)

        for action in skill_controller:
            next_obs, reward, done, info = self.env.step(action)
            self.step_count += 1

            step_data = {}
            step_data["obs"] = curr_obs_processed
            step_data["action"] = action
            step_data["reward"] = reward
            step_data["done"] = done

            next_obs_processed = process_observation(next_obs)
            step_data["next_obs"] = next_obs_processed

            self.current_traj_histories.append(step_data)

            self.curr_obs = next_obs
            curr_obs_processed = next_obs_processed

        if len(self.current_traj_histories) > 0:
            self.flush_current_traj(skill_type)
            self.current_traj_histories = []
            self.traj_count += 1

    def step(self, action):
        """
        Run the environment step() function and collect data
        Do not use this function for skill execution (use execute_skill instead)

        Args:
            action (np.array): action to take in environment

        Returns:
            4-tuple:
                - (dict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        obs_data = process_observation(self.curr_obs)
        next_obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        step_data = {}
        step_data["obs"] = obs_data
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["next_obs"] = process_observation(next_obs)
        step_data["done"] = done
        self.current_traj_histories.append(step_data)

        self.curr_obs = next_obs

        return next_obs, reward, done, info

    def reset(self):
        """
        Run the environment reset() function and flush data

        Returns:
            dict: Environment observation space after reset occurs
        """
        if len(self.current_traj_histories) > 0:
            self.flush_current_traj()
            self.current_traj_histories = []
            self.traj_count += 1

        self.curr_obs = self.env.reset()
        return self.curr_obs

    def observation_spec(self):
        """
        Grab the normal environment observation_spec

        Returns:
            dict: Observations from the environment
        """
        return self.env.observation_spec()
    
    def flush_current_traj(self, skill_type=None):
        """
        Flush current trajectory data
        """
        process_traj_to_hdf5(self.current_traj_histories, self.hdf5_file, skill_type=skill_type)

    def save_data(self):
        """
        Save collected trajectories as a hdf5 file in the robomimic format

        Args:
            path (str): path to store robomimic hdf5 data file
        """
        self.hdf5_file["group"].attrs["total"] = self.step_count
        self.hdf5_file.close()
