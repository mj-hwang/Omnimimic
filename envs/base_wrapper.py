from omnigibson.envs.env_wrapper import EnvironmentWrapper
from robomimic.envs.env_base import EnvType
from utils.env_util import *
import json
import h5py

class OmniRobomimicWrapper(EnvironmentWrapper):
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
        
        self.curr_obs = None
        self.current_traj_data = []
        self.hdf5_file = h5py.File(path, 'w')
        self.data_grp = self.hdf5_file.create_group(f"data")
        
        # TODO: update env kwargs
        self.env_args = {
            "env_name": self.env.name,
            "env_type": EnvType.GYM_TYPE,
            "env_kwargs": {},
        }
        self.data_grp.attrs["env_args"] = json.dumps(self.env_args)

        # Run super
        super().__init__(env=env)

    def step(self, action):
        """
        By default, run the normal environment step() function

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
        self.current_traj_data.append(step_data)

        self.curr_obs = next_obs

        return next_obs, reward, done, info

    def reset(self):
        """
        By default, run the normal environment reset() function

        Returns:
            dict: Environment observation space after reset occurs
        """
        if len(self.current_traj_histories) > 0:
            self.flush_data()
            self.current_traj_histories = []
            self.traj_count += 1

        self.curr_obs = self.env.reset()
        return self.curr_obs

    def observation_spec(self):
        """
        By default, grabs the normal environment observation_spec

        Returns:
            dict: Observations from the environment
        """
        return self.env.observation_spec()
    
    def flush_current_traj(self):
        """
        Flush current trajectory data
        """
        process_traj_to_hdf5(self.current_traj_histories, self.data_grp)

    def save_data(self):
        """
        Save collected trajectories as a hdf5 file in the robomimic format

        Args:
            path (str): path to store robomimic hdf5 data file
        """
        self.data_grp.attrs["total"] = self.step_count
        self.hdf5_file.close()
