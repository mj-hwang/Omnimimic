import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.envs.env_wrapper import EnvironmentWrapper
import h5py

class OmniSkillWrapper(EnvironmentError):
    """
    Base class for all environment wrappers in OmniGibson. In general, reset(), step(), and observation_spec() should
    be overwritten

    Args:
        env (OmniGibsonEnv): The environment to wrap.
    """

    def __init__(self, env):
        self.env = env

        # Run super
        super().__init__(obj=env)

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
        return self.step(action)

    def reset(self):
        """
        By default, run the normal environment reset() function

        Returns:
            dict: Environment observation space after reset occurs
        """
        return self.reset()
    
    def execute_controller(ctrl_gen, env, skill_type):
        # If no action is given, do nothing
        if len(ctrl_gen) == 0:
            return []
        
        oa_skill = []
        # Iterate through low-level actions
        for action in ctrl_gen:
            obs, rew, done, info = env.step(action)

            # Check if this action is in "navigation phase"
            base_action = action[env.robots[0].robot.controller_action_idx["base"]]
            if max(base_action) < 1e-8:
                skill_type = skill_type + "_nav"

            # Process observation and action
            oa_data = process_observation_and_action(env, obs, action, skill_type=skill_type)
            oa_skill.append(oa_data)
            if done:
                og.log.info("Episode finished")
                break
        
        # Update "skill_done" and "done" accordingly
        oa_skill[-1]["skill_done"] = True
        if done:
            oa_skill[-1]["done"] = False

        return oa_skill
