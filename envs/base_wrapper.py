from omnigibson.envs.env_wrapper import EnvironmentWrapper
from robomimic.envs.env_base import EnvType
from utils.env_util import *
import json
import h5py

h5py.get_config().track_order = True

class OmnimimicBaseWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for collecting data in robomimic format. 

    Args:
        env (OmniGibsonEnv): The environment to wrap.
    """
    def __init__(self, env, obs_modalities, path):
        self.env = env
        self.obs_modalities = obs_modalities

        self.traj_count = 0
        self.step_count = 0
        
        self.current_ibs = None
        self.current_traj_histories = []
        self.hdf5_file = h5py.File(path, 'w')
        self.hdf5_file.create_group("data")
        self.hdf5_file.create_group("mask")
        
        # TODO: update env name and kwargs
        self.env_args = {
            "env_name": "omni_test",
            "env_type": EnvType.GYM_TYPE,
            "env_kwargs": {},
        }
        self.hdf5_file["data"].attrs["env_args"] = json.dumps(self.env_args)

        # Run super
        super().__init__(env=env)
        self.reset()

    def step(self, action):
        """
        Run the environment step() function and collect data

        Args:
            action (np.array): action to take in environment

        Returns:
            4-tuple:
                - (dict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        obs_data = process_observation(self.env, self.current_obs, self.obs_modalities)
        next_obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        step_data = {}
        step_data["obs"] = obs_data
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["next_obs"] = process_observation(self.env, next_obs, self.obs_modalities)
        step_data["done"] = done
        self.current_traj_histories.append(step_data)

        self.current_obs = next_obs

        return next_obs, reward, done, info

    def reset(self):
        """
        Run the environment reset() function and flush data

        Returns:
            dict: Environment observation space after reset occurs
        """
        if len(self.current_traj_histories) > 0:
            self.flush_current_traj()

        self.current_obs = self.env.reset()
        return self.current_obs

    def observation_spec(self):
        """
        Grab the normal environment observation_spec

        Returns:
            dict: Observations from the environment
        """
        return self.env.observation_spec()
    
    def flush_current_traj(self):
        """
        Flush current trajectory data
        """
        traj_grp_name = f"demo_{self.traj_count}"
        process_traj_to_hdf5(self.current_traj_histories, self.hdf5_file, traj_grp_name)
        self.current_traj_histories = []
        self.traj_count += 1

    def save_data(self):
        """
        Save collected trajectories as a hdf5 file in the robomimic format

        Args:
            path (str): path to store robomimic hdf5 data file
        """
        self.hdf5_file["data"].attrs["total"] = self.step_count
        self.hdf5_file.close()
