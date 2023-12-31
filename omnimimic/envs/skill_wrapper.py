from omnigibson.envs.env_wrapper import EnvironmentWrapper
from robomimic.envs.env_base import EnvType
from collections import defaultdict
import json
import h5py

from omnimimic.utils.env_util import *
from omnimimic.utils.robot_util import *
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError

h5py.get_config().track_order = True

class OmnimimicSkillWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for collecting data in robomimic format,
    when using action primitive skills developed in Omniverse.

    Note: current "skill" history stores transitions during an execution of a specific skill,
    whereas current "traj" history stores transitions until the termination of a trajectory.
    More specifically, each element in current_traj_history is a tuple of skill type and skill history.

    Args:
        env (OmniGibsonEnv): The environment to wrap.
        obs_modalities (list): list of observation modalities to collect
        path (str): path to store robomimic hdf5 data file
        collect_partial (bool): whether to collect partial trajectories when it errors or fails before completion
        collect_if_success_at_least_once (bool): whether to collect a trajectory if it succeeds at least once after skill
        collect_if_success_at_the_end (bool): whether to collect a trajectory if it succeeds at the end
        save_per_skill (bool): If True, each skill executed via execute_skill is saved as a separate trajectory.
            If false, the entire task demonstration is saved as a single trajectory once the task succeeds.
    """
    def __init__(
        self, 
        env, 
        obs_modalities, 
        path, 
        collect_partial=False,
        collect_if_success_at_least_once=True,
        collect_if_success_at_the_end=False,
        use_delta=True,
        save_per_skill=False,
        save_next_obs=False,
    ):
        self.env = env
        self.obs_modalities = obs_modalities

        self.collect_partial = collect_partial
        self.collect_if_success_at_least_once = collect_if_success_at_least_once
        self.collect_if_success_at_the_end = collect_if_success_at_the_end

        self.traj_count = 0
        self.step_count = 0
        self.env_traj_count = 0
        self.manipulation_traj_count = 0
        self.navigation_traj_count = 0
        
        self.current_obs = None
        self.current_traj_history = []
        self.current_skill_history = [] 
        self.ever_succeededed = False
        self.save_per_skill = save_per_skill
        self.save_next_obs = save_next_obs

        self.skill_mask_dict = defaultdict(list)
        self.env_args = self.env.env_args

        self.data_path = path
        with h5py.File(self.data_path, 'w') as f:
            data_grp = f.create_group("data")
            f.create_group("mask")
            data_grp.attrs["env_args"] = json.dumps(self.env_args)

        self.control_limits = get_control_limits(self.env.robot)

        # Run super
        super().__init__(env=env)
        self.reset()

    def execute_skill(self, skill_generator, skill_name=None):
        """
        Execute the skill given the skill controller.
        If planning or execution fails, trajectory is discarded.

        Note: skill_type can be different from skill_name.
        Most manipulation skills in OmniGibson have two phases: navigation and manipulation,
        and we indicate the skill type by appending "_nav" or "_manip" to the skill name.

        Args:
            skill_generator (Generator): a generator of action to execute the skill
            skill_name (str): name of the skill
        """
        if self.save_per_skill:
            assert skill_name is not None, "Skill name must be provided to save per-skill trajectories"
        try:
            print("[1] current_skill_history len", len(self.current_skill_history))
            current_obs_processed = process_omni_obs(self.current_obs, self.obs_modalities)

            current_skill_type = None
            # self.current_skill_history = []
            for action, skill_info in skill_generator:

                # if saving per skill trajectories, keep track of skill info 
                if self.save_per_skill:
                    skill_type = skill_info.split(":")[0]
                    if skill_type == "nav":
                        skill_type = skill_name + "_nav"
                    elif skill_type == "manip":
                        skill_type = skill_name + "_manip"
                    else:
                        # Some actions (e.g. _settle_robot) have "idle" type.
                        # These are considered to be a continuation of the current skill type.
                        skill_type = current_skill_type

                    if skill_type != current_skill_type:
                        if current_skill_type is not None and len(self.current_skill_history) > 0:
                            self.current_traj_history.append(
                                (current_skill_type, self.current_skill_history)
                            )
                        current_skill_type = skill_type
                        self.current_skill_history = []

                # record step data
                step_data = {}
                step_data["action"] = normalize_action(action.copy(), self.control_limits)

                next_obs, reward, done, info = self.env.step(action)
                
                step_data["obs"] = current_obs_processed
                step_data["reward"] = reward
                step_data["done"] = done

                next_obs_processed = process_omni_obs(next_obs, self.obs_modalities)
                if self.save_next_obs:
                    step_data["next_obs"] = next_obs_processed

                self.current_skill_history.append(step_data)

                self.current_obs = next_obs
                current_obs_processed = next_obs_processed

                if done:
                    break  
            # print("[2] current_skill_history len", len(self.current_skill_history))
            if self.save_per_skill:
                if len(self.current_skill_history) > 0:
                    self.current_traj_history.append(
                        (current_skill_type, self.current_skill_history)
                    )
                    self.current_skill_history = []

            if self.env.is_success():
                if not self.save_per_skill:
                    self.current_traj_history.append(
                        (current_skill_type, self.current_skill_history)
                    )
                    self.current_skill_history = []
                self.ever_succeededed = True
            # print("[3]current_skill_history len", len(self.current_skill_history))
            # print("current traj history len", len(self.current_traj_history)) 
            # breakpoint()
        except ActionPrimitiveError as err:
            print("skill execution failed", err)
            if not self.collect_partial:
                self.current_traj_history = []
                self.current_skill_history = []


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
        if len(self.current_traj_history) > 0:
            if self.collect_if_success_at_the_end:
                if self.env.is_success():
                    self.flush_current_traj()
                else:
                    self.current_traj_history = []

            elif self.collect_if_success_at_least_once:
                if self.ever_succeededed:
                    self.flush_current_traj()
                else:
                    self.current_traj_history = []
            else:
                print("flushing")
                self.flush_current_traj()

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
            # append current traj history to hdf5 file
            print(f"\n\n{len(self.current_traj_history)} trajctories will be flushed")
            # if not (len(self.current_traj_history) == 3):
            #     breakpoint()
            for skill_type, skill_history in self.current_traj_history:
                traj_grp_name = f"demo_{self.traj_count}"
                process_traj_to_hdf5(skill_history, f, traj_grp_name, save_next_obs=self.save_next_obs)
                if self.save_per_skill:
                    self.skill_mask_dict[skill_type].append(traj_grp_name)
                    if "nav" in skill_type:
                        self.navigation_traj_count += 1
                    else:
                        self.manipulation_traj_count += 1

                self.traj_count += 1
                self.step_count += len(skill_history)

            self.env_traj_count += 1
            self.current_traj_history = []
            self.current_skill_history = []

            # update total step and skill mask in hdf5 file
            f.require_group("data").attrs["total"] = self.step_count
            
            if self.save_per_skill:
                for skill_type, grps in self.skill_mask_dict.items():
                    mask_grp = f.require_group("mask")
                    if skill_type in mask_grp:
                        del mask_grp[skill_type]
                    mask_grp.create_dataset(skill_type, data=grps)

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