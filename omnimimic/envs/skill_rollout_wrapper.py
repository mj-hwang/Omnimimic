from omnigibson.envs.env_wrapper import EnvironmentWrapper
import robomimic.envs.env_base as EB
from collections import defaultdict
import json
import h5py
import numpy as np

from omnimimic.utils.env_util import *
from omnimimic.utils.robot_util import *
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError

h5py.get_config().track_order = True

class OmnimimicSkillRolloutWrapper(EB.EnvBase):
    """
    An OmniGibson environment wrapper for rolloing out policy
    when using action primitive skills developed in Omniverse.
    
    Note: current "skill" history stores transitions during an execution of a specific skill,
    whereas current "traj" history stores transitions until the termination of a trajectory.
    More specifically, each element in current_traj_history is a tuple of skill type and skill history.

    Args:
        env (OmniGibsonEnv): The environment to wrap.
        obs_modalities (list): list of observation modalities to collect
        path (str): path to hdf5 data file

    NOTE: Differences from OmnimimicSkillWrapper:
    - DOES NOT automatically flush trajectories to datafile with skill change or reset
    
    """
    def __init__(
        self, 
        env, 
        obs_modalities, 
        path, 
    ):
        self.env = env
        self.obs_modalities = obs_modalities

        self.data_path = path
        with h5py.File(self.data_path, 'r+') as f:
            # make sure dataset has expected structure
            assert "data" in f.keys(), "Dataset is missing 'data' group"
            assert "mask" in f.keys(), "Dataset is missing 'mask' group"
            assert "env_args" in f["data"].attrs.keys(), "Dataset is missing 'env_args' attribute"

            # get number of trajectories and total steps in original dataset
            self.traj_count = len(f["data"].keys())
            self.step_count = f["data"].attrs["total"]
        
        # self.current_obs = None
        self.current_traj_history = []
        self.ever_succeededed = False

        self.skill_mask_dict = defaultdict(list)
        self.env_args = self.env.env_args
        self.control_limits = get_control_limits(self.env.robot)

        self.current_skill_type = None

        self.current_obs = self.env.reset()

    def execute_skill(self, skill_generator, skill_name, video_writer=None, video_skip=5):
        """
        Execute the skill given the skill controller.
        If planning or execution fails, trajectory is discarded.

        Note: skill_type can be different from skill_name.
        Most manipulation skills in OmniGibson have two phases: navigation and manipulation,
        and we indicate the skill type by appending "_nav" or "_manip" to the skill name.

        Args:
            skill_generator (Generator): a generator of action to execute the skill
            skill_name (str): name of the skill
            video_writer (imageio Writer instance): if None, append frames to video writer at rate @video_skip
            video_skip (int): number of frames to skip when writing to video

        Returns:
            results (dict): dictionary of results from the rollout
            rollout_success (bool): whether the rollout successfully completed the task
                if rollout was succes, trajectory is flushed to datafile
        """
        n_steps = 0

        results = {}
        video_count = 0  # video frame counter

        total_reward = 0.
        success = { k: False for k in self.is_success() } # success metrics

        try:
            current_obs_processed = process_omni_obs(self.current_obs, self.obs_modalities)

            current_skill_type = None
            current_skill_history = []
            for action, skill_info in skill_generator:
                skill_type = skill_info.split(":")[0]
                if skill_type == "nav":
                    skill_type = skill_name + "_nav"
                else:
                    skill_type = skill_name + "_manip"
                if skill_type != current_skill_type:
                    print(f"\nskill type changed from {current_skill_type} to {skill_type}\n")
                    if current_skill_type is not None and len(current_skill_history) > 0:
                        self.current_traj_history.append(
                            (current_skill_type, current_skill_history)
                        )
                    current_skill_type = skill_type
                    current_skill_history = []

                next_obs, reward, done, info = self.env.step(action)
                n_steps += 1

                step_data = {}
                step_data["obs"] = current_obs_processed
                step_data["action"] = normalize_action(action, self.control_limits)
                step_data["reward"] = reward
                step_data["done"] = done

                next_obs_processed = process_omni_obs(next_obs, self.obs_modalities)
                step_data["next_obs"] = next_obs_processed

                current_skill_history.append(step_data)

                self.current_obs = next_obs
                current_obs_processed = next_obs_processed

                total_reward += reward
                cur_success_metrics = self.is_success()
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

                # visualization - # TODO: figure out offscreen rendering in og env (implement render function in moma_wrapper)
                if video_writer is not None:
                    if video_count % video_skip == 0:
                        video_img = self.env.render(mode="rgb_array", height=512, width=512)
                        video_writer.append_data(video_img)

                    video_count += 1

            if len(current_skill_history) > 0:
                self.current_traj_history.append(
                    (current_skill_type, current_skill_history)
                )

            # check task success
            if self.is_success()["task"]:
                # self.flush_current_traj()
                rollout_success = True
            else:
                rollout_success = False

        except ActionPrimitiveError as err:
            print("skill execution failed", err)
            rollout_success = False


        results["Return"] = total_reward
        results["Horizon"] = n_steps + 1
        results["SuccessRate"] = float(success["task"])

        return results, rollout_success

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
        return self.env.step(action)

    def reset(self):
        """
        Run the environment reset() function and flush data

        Returns:
            dict: Environment observation space after reset occurs
        """
        self.current_traj_history = []

        self.current_obs = self.env.reset()
        self.ever_succeededed = False
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
        with h5py.File(self.data_path, 'r+') as f:
            breakpoint()
            # append current traj history to hdf5 file
            for skill_type, skill_history in self.current_traj_history:
                traj_grp_name = f"demo_{self.traj_count}"
                process_traj_to_hdf5(skill_history, f, traj_grp_name)
                self.skill_mask_dict[skill_type].append(traj_grp_name)
                
                self.traj_count += 1
                self.step_count += len(skill_history)

            self.current_traj_history = []

            # update total step and skill mask in hdf5 file
            f.require_group("data").attrs["total"] = self.step_count
            for skill_type, grps in self.skill_mask_dict.items():
                mask_grp = f.require_group("mask")
                if skill_type in mask_grp:
                    del mask_grp[skill_type]
                mask_grp.create_dataset(skill_type, data=grps)
    
    # def clear_current_traj(self):
    #     """
    #     Clears current trajectory history
    #     """
    #     self.current_traj_history = []

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

    def get_current_traj_history(self):
        return self.current_traj_history

    ########## env_util functions for easy access ##########
    def normalize_action(self, action):
        return normalize_action(action, self.control_limits)
    
    def denormalize_action(self, action):
        return denormalize_action(action, self.control_limits)

    def process_obs(self, obs, postprocess_for_eval=False):
        return process_omni_obs(obs, self.obs_modalities, postprocess_for_eval=postprocess_for_eval)
    
    def process_traj_to_datafile(traj_data, hdf5_file, traj_grp_name):
        return process_traj_to_hdf5(traj_data, hdf5_file, traj_grp_name)
    
    ########## functions needed for robomimic EnvBase type ##########
    def reset_to(self, state): # TODO?
        print("reset_to not implemented")
        return
    
    def render(self):
        return

    def get_observation(self, obs=None, postprocess_for_eval=False):
        """
        Reshape observations to shapes expected by robomimic

        RGB (H, W, 3) --> (3, H, W) 
        depth (H, W) --> (1, H, W) 
        proprio (N,) --> (N,)
        scan (M, 1) --> (1, M)
        """
        if obs is None:
            obs = self.obs
        return process_omni_obs(obs, self.obs_modalities, postprocess_for_eval=postprocess_for_eval)

    def get_state(self):
        return self.current_obs
    
    def get_reward(self):
        raise NotImplementedError()
    
    @classmethod
    def get_goal(self):
        raise NotImplementedError()
    
    def set_goal(self):
        raise NotImplementedError()
     
    def is_done(self):
        return self.env.is_done()
    
    def is_success(self):
        return {"task" : self.env.is_success()}
    
    @property
    def action_dimension(self):
        return self.robot.action_dim
    
    @property
    def name(self):
        return self.env.name
    
    @property
    def type(self):
        return EB.EnvType.OMNIMIMIC_TYPE
    
    def serialize(self):
        return self.env_args
    
    @classmethod
    def create_for_data_processing(cls, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        return

    @property
    def rollout_exceptions(self):
        """Return tuple of exceptions to except when doing rollouts"""
        return (RuntimeError)








