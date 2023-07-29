from omnigibson.envs.env_wrapper import EnvironmentWrapper
from robomimic.envs.env_base import EnvType
from utils.env_util import *
from collections import defaultdict
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

        self.manipulation_traj_count = 0
        self.navigation_traj_count = 0
        
        self.current_obs = None
        self.current_skill_type = None

        self.current_traj_histories = []
    
        self.hdf5_file = h5py.File(path, 'w')
        self.hdf5_file.create_group("data")
        self.hdf5_file.create_group("mask")

        self.skill_mask_dict = defaultdict(list)
        
        # TODO: update env kwargs
        self.env_args = {
            "env_name": "omni_test",
            "env_type": EnvType.GYM_TYPE,
            "env_kwargs": {},
        }
        self.hdf5_file["data"].attrs["env_args"] = json.dumps(self.env_args)

        # Run super
        super().__init__(env=env)
        self.reset()

    def execute_skill(self, skill_controller, skill_name):
        """
        Execute the skill given the skill controller, and flush data
        Data is flushed every time skill type changes.
        If planning or execution fails, trajectory is discarded.
        """
        current_obs_processed = process_omni_obs(self.current_obs, self.obs_modalities)

        try:
            for action in skill_controller:
                skill_type = get_skill_type(self.env, action, skill_name)
                print("skill type", skill_type)
                if skill_type != self.current_skill_type:
                    if self.current_skill_type is not None and len(self.current_traj_histories) > 0:
                        # self.flush_current_traj(skill_type)
                        self.flush_current_traj()
                    self.current_skill_type = skill_type

                next_obs, reward, done, info = self.env.step(action)
                self.step_count += 1

                step_data = {}
                step_data["obs"] = current_obs_processed
                step_data["action"] = action
                step_data["reward"] = reward
                step_data["done"] = done

                next_obs_processed = process_omni_obs(next_obs, self.obs_modalities)
                step_data["next_obs"] = next_obs_processed

                self.current_traj_histories.append(step_data)

                self.current_obs = next_obs
                current_obs_processed = next_obs_processed

            if len(self.current_traj_histories) > 0:
                self.flush_current_traj()
        
        except:
            pass

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
        obs_data = process_omni_obs(self.current_obs)
        next_obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        step_data = {}
        step_data["obs"] = obs_data
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["next_obs"] = process_omni_obs(next_obs)
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
        Flush current trajectory data and update mask for skill type
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

        for skill_type, grps in self.skill_mask_dict.items():
            self.hdf5_file["mask"].create_dataset(skill_type, data=grps)

        self.hdf5_file.close()
