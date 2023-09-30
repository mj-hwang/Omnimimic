import yaml
import numpy as np
import json
import os

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import OnTop
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.utils.config_utils import NumpyEncoder

from omnimimic.utils.macros import gm as omni_gm

class MOMAEnv():
    
    """
    Custom environment wrapper for distilling-moma project.
    Compatible with Omnigibson action-primitives branch as of 9/29/2023
    
    Args:
        config_filename (str): path to config file
        max_steps (int): maximum number of steps before termination
        random_init_base_pose (None or list): if None, initialize robot base at default pose or pose set by initial_robot_pose.
            if list, initialize robot at random pose within range specified by list ([x_low, x_high], [y_low, y_high], [yaw_low, yaw_high])
        random_init_arm_pose (dict): Maps arm (str) to whether to randomize (bool) e.g. {"left": True, "right" : False}. DOES NOT CHECK FOR SELF COLLISION
        initial_robot_pose (list): default initial base pose [x, y, yaw]. Only used if @random_init_base_pose is not set
        raise_hand (bool): if True, raise hand at start of episode and set the tuck pose to raised hand pose
        rigid_trunk (bool) : if True, maintain trunk position to default_trunk_offfset defined in robot config
        localization_noise (dict): gaussian noise to apply to robot pose in world observation (x, y, yaw)
            {"mean" : (x mean, y mean, yaw mean), "std" : (x std, y std, yaw std)} # NOTE - not implemented yet
        simple_scene (bool) : if true, only load floor, wall, and table
    """
    
    def __init__(
        self,
        env_name,
        config_filename,
        max_steps=1000,
        random_init_base_pose=None, # randomly initialize robot base pose
        random_init_arm_pose=None, # randomly initialize arm poses 
        initial_robot_pose=[0.0, 0.0, 0.0], # defualt initial pose [x, y, yaw]
        raise_hand=True,
        rigid_trunk=False,
        localization_noise={ # TODO - not implemented yet
            "mean" : (0.0, 0.0, 0.0),
            "std" : (0.0, 0.0, 0.0),
        }, # gaussian noise to apply to robot base pose observation (x, y, yaw)
        simple_scene=False,
        render=False, # NOTE: not used. only here for robomimic compatibility
        render_offscreen=False, # NOTE: not used. only here for robomimic compatibility
        use_image_obs=False, # NOTE: not used. only here for robomimic compatibility
        postprocess_visual_obs=False, # NOTE: not used. only here for robomimic compatibility
    ):
        # Load the config
        config_path = os.path.join(omni_gm.ROOT_DIR, "examples/example_custom_envs/configs", config_filename)
        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        if simple_scene:
            self.config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]
        # Randomization settings
        self.random_init_base_pose = random_init_base_pose
        self.random_init_arm_pose = random_init_arm_pose

        # Load the environment
        self.rigid_trunk = rigid_trunk
        if self.rigid_trunk:
            assert self.config["robots"][0]["type"] == "Tiago", "rigid trunk only works for Tiago"
            self.config["robots"][0]["rigid_trunk"] = True

        self.env = og.Environment(action_timestep=1/10.0, configs=self.config)
        self.scene = self.env.scene
        self.robots = self.env.robots
        self.robot = self.env.robots[0]
        self.robot_model = self.robot.model_name

        # other settings
        self.name = env_name
        self.raise_hand = raise_hand
        self.initial_robot_pose = initial_robot_pose

        self.localization_noise = localization_noise

        # initialize primitive skill controller
        self.controller = StarterSemanticActionPrimitives(
            task=None,
            scene=self.scene,
            robot=self.robot,
            add_context=True,
        )

        self.max_steps = max_steps

        # Allow user to move camera more easily
        og.sim.enable_viewer_camera_teleoperation()

        # store objects
        self.objects = {} # everything in the scene
        for obj in self.scene.objects:
            self.objects[obj.name] = obj
        self.unique_objects = {} # objects unique to this environment
        
        # set robot to initial configuration
        # arm joint limits (used for random initialization)
        self.arm_joint_upper_limits = {
            "left" : [],
            "right" : [],
        }
        self.arm_joint_lower_limits = {
            "left" : [],
            "right" : [],
        }
        for i in range(1, 8):
            self.arm_joint_upper_limits["left"].append(self.robot.joints[f"arm_left_{i}_joint"].upper_limit)
            self.arm_joint_upper_limits["right"].append(self.robot.joints[f"arm_right_{i}_joint"].upper_limit)
            self.arm_joint_lower_limits["left"].append(self.robot.joints[f"arm_left_{i}_joint"].lower_limit)
            self.arm_joint_lower_limits["right"].append(self.robot.joints[f"arm_right_{i}_joint"].lower_limit)
            
        self._initialize_robot_pose()

        self.steps = 0

        # keep track of gripper actions (0 = keep at same position, -1 = close, 1 = open)
        self.prev_gripper_action = 0.0
        self.conseq_gripper_action_cnt = 0
        self.gripper_state = 0.0
        self.gripper_filter_horizon = 5 # how many steps the same action must be received to execute gripper action
    
    def restore(self, og_load_path, moma_load_path):
        """
        Restores simulator state to @og_load_path and environment state to @moma_load_path

        Args:
            og_load_path (str) : Full path of JSON file to load (should end with .json), which contains information
                to recreate the current scene. This is the file that will be used to load the scene.
            moma_load_path (str) : Full path of JSON file to load (should end with .json), which contains information
                specific to this environment (e.g. custom objects that are not a part of the default scene, robot info, etc.)
        """
        # Call og's restore function
        og.sim.restore(og_load_path)

        # update scene and robot to newly restored ones
        self.scene = self.env.scene
        self.robots = self.env.robots
        self.robot = self.env.robots[0]

        # update objects
        for obj in self.scene.objects:
            if obj.name in self.unique_objects.keys():
                self.unique_objects[obj.name] = obj
            self.objects[obj.name] = obj

        # reinitialize skill generator
        self.controller = StarterSemanticActionPrimitives(
            task=None,
            scene=self.scene,
            robot=self.robot,
        )

        # Reset the environment
        obs = self._reset_to(moma_load_path)

        return obs

    def _reset_to(self, json_path):
        """
        Resets environment to state specified by @json_path

        Args:
            json_path (str) : Full path of JSON file to load (should end with .json), which contains information
                to recreate the current scene. This is the file that will be used to load the scene.
        """

        # load json
        with open(json_path, "r") as f:
            env_info = json.load(f)

        # get updated observations
        obs = {
            "task" : {
                "low_dim" : np.array([], dtype=np.float64),
            }
        }
        obs["robot0"] = self.robot.get_obs()

        # take a few steps
        self.step_sim(5)

        # reset steps counter
        self.steps = env_info["current_step"]

        return obs


    def _place_objects(self):
        """
        Set poses of custom objects in this environment. Should be overwritten in child class.
        """
        return

    def add_objects(self):
        """
        Adds custom objects to environment. Should be overwritten in child class.
        """
        return

    def save(self, og_save_path, moma_save_path):
        """
        Saves the current simulation environment to @json_path.

        Args:
            og_save_path (str): Full path of JSON file to save (should end with .json), which contains information
                to recreate the current scene. This is the file that will be used to load the scene.
            moma_save_path (str): Full path of JSON file to save (should end with .json), which contains information
                specific to this environment (e.g. custom objects that are not a part of the default scene, robot info, etc.)
        """
        # Make sure the sim is not stopped, since we need to grab joint states
        assert not og.sim.is_stopped(), "Simulator cannot be stopped when saving to USD!"

        # Update scene info
        self.scene.update_objects_info()

        scene_info = {
            "current_step" : self.steps,
        }

        # Save info to recreate scene at og level
        og.sim.save(os.path.join("env_snapshots/og_scene", og_save_path))

        # Save info specific to this env
        with open(os.path.join("env_snapshots/moma_env", moma_save_path), "w+") as f:
            json.dump(scene_info, f, cls=NumpyEncoder, indent=4)


    def _initialize_robot_pose(self):

        self.robot.set_joint_positions(self.default_joint_pos)
        
        # take a few steps
        self.step_sim(10)

        # set base pose
        if self.random_init_base_pose is not None:
            x_lim = self.random_init_base_pose[0]
            y_lim = self.random_init_base_pose[1]
            yaw_lim = self.random_init_base_pose[2]
            x = np.random.uniform(low=x_lim[0], high=x_lim[1], size=1)[0]
            y = np.random.uniform(low=y_lim[0], high=y_lim[1], size=1)[0]
            yaw = np.random.uniform(low=yaw_lim[0], high=yaw_lim[1], size=1)[0]
            
            self.robot.set_position([x, y, 0.05])
            quat = T.euler2quat([0.0, 0.0, yaw])
            self.robot.set_orientation(quat)
            print("initialized robot at (pos, yaw, quat)\n", x, y, yaw, quat)

        else:
            pos = [self.initial_robot_pose[0], self.initial_robot_pose[1], 0.05]
            self.robot.set_position(pos)
            quat = T.euler2quat([0.0, 0.0, self.initial_robot_pose[2]])
            self.robot.set_orientation(quat)
            print("initialized robot at (pos, yaw, quat)\n", pos, quat)
        
        # set arm pose
        if self.random_init_arm_pose is not None:
            for arm in self.random_init_arm_pose:
                if self.random_init_arm_pose[arm]:

                    # add noise to default joint positions
                    control_idx = self.robot.arm_control_idx[arm]
                    default_q = self.default_joint_pos[control_idx]
                    noise = np.random.uniform(low=-0.5, high=0.5, size=control_idx.shape[0])
                    arm_q = default_q + noise
                    self.robot.set_joint_positions(arm_q, control_idx)

        # take a few steps
        self.step_sim(10)

    def add_to_objects_dict(self, objects):
        """
        Add objects to self.objects dict and self.unique_objects dict
        Args:   
            objects (list): list of objects to add to self.objects
        """
        for obj in objects:
            self.objects[obj.name] = obj
            self.unique_objects[obj.name] = obj

    def check_termination(self):
        """
        Check if termination condition has been met. Overwrite this in child class.

        Returns:
            done (bool): True if termination condition has been met
            info (dict): dictionary of termination conditions
        """
        raise NotImplementedError()
    
    def _check_horizon_reached(self):
        return self.steps >= self.max_steps
    
    def _filter_gripper_action(self, action):
        """
        Filter gripper action to prevent rapid opening and closing of gripper
        """
        if self.robot_model == "Tiago":
            gripper_action = action[self.robot.controller_action_idx["gripper_" + self.robot.default_arm]]
            # print("input action", action)
            if gripper_action > 0.0:
                gripper_action = 1.0
            elif gripper_action < 0.0:
                gripper_action = -1.0

            if (not gripper_action == 0) and (gripper_action == self.prev_gripper_action):
                self.conseq_gripper_action_cnt += 1
            else:
                self.conseq_gripper_action_cnt = 0

            self.prev_gripper_action = gripper_action

            if self.conseq_gripper_action_cnt >= self.gripper_filter_horizon:
                self.gripper_state = gripper_action
                return gripper_action
            return 0.0

    def reward(self):
        return 0

    def step(self, action):

        # apply gripper action filter
        gripper_action = self._filter_gripper_action(action)
        action[self.robot.controller_action_idx["gripper_" + self.robot.default_arm]] = gripper_action
        
        obs, _, _, _ = self.env.step(action) # only use obs from environment - reward, done, info comes from task, but env.task is DummyTask

        # get reward
        reward = self.reward() # returns 0 by default

        # get done and info
        done, info = self.check_termination()
        
        self.steps += 1
        return obs, reward, done, info

    def step_sim(self, steps):
        """
        Steps simulation without taking action
        """
        for _ in range(steps):
            og.sim.step()

    def reset(self):
        
        # reset environment
        obs = self.env.reset()

        # refresh sim
        og.sim.stop()
        og.sim.play()

        # reset robot pose
        self.tuck_arm()
        self._initialize_robot_pose()

        # take a few steps 
        self.step_sim(10)

        self.steps = 0

        # get most recent observation
        obs["robot0"] = self.robot.get_obs()
        return obs

    def _check_ontop(self, objA, objB):
        """
        Checks if objA is on top of objB
        """
        # print("-------check_ontop---------")

        return objA.states[OnTop]._get_value(objB)
    
    def _is_gripper_closed(self):
        """
        Checks if gripper is closed
        Args:
            arm (str): "left" or "right" - only works for Tiago right now. not used
        """
        # print("-------is_gripper_closed---------")

        return self.robot.controllers["gripper_" + self.robot.default_arm].is_grasping()
    
    def _is_grasping_obj(self, obj_name):
        """
        Checks if gripper is grasping obj
        """
        # print("-------is_grasping_obj---------")

        obj = self.robot._ag_obj_in_hand[self.robot.default_arm]
        if obj is None:
            return False
        
        return obj.name == obj_name

    def _is_obj_on_floor(self, obj):
        """
        Checks if obj is on floor
        """
        # print("-------is_obj_on_floor---------")
        floor_objects = [self.objects[obj_name] for obj_name in self.objects if "floor" in obj_name]
        for floor_obj in floor_objects:
            if self._check_ontop(obj, floor_obj):
                return True
        return False
    
    def _teleport_robot(self, pose):
        """
        Teleports robot to specified pose
        Args:
            pose (list): [x, y, yaw]
        """
        quat = T.euler2quat([0.0, 0.0, pose[2]])
        self.robot.set_position([pose[0], pose[1], 0.05])
        self.robot.set_orientation(quat)
        self.step_sim(5)

    def open_grippers(self):
        """
        Sets gripper joint to open position
        """
        release_action = self.controller._empty_action()
        release_action[self.robot.controller_action_idx["gripper_left"]] = 1.0
        for _ in range(50):
            self.env.step(release_action)

    def tuck_arm(self):
        if self.raise_hand:
            self.robot.set_joint_positions(self.default_joint_pos)
        else:
            self.robot.tuck()
        # self.step_sim(10)

    def _is_arm_homed(self, arm="all"):
        """
        Checks if arm is in home position.
        Args:
            arm: "left", "right" or "all"
        """
        if self.robot_model == "Tiago":
            if arm == "all":
                control_idx = np.concatenate([self.robot.arm_control_idx["left"], self.robot.arm_control_idx["right"]])
            else:
                control_idx = self.robot.arm_control_idx[arm]        
        
        elif self.robot_model == "Fetch":
            control_idx = self.robot.arm_control_idx[self.robot.default_arm]

        thresh = 0.05
        # print(max(abs(self.robot.get_joint_positions() - self.default_joint_pos)[control_idx]))
        return max(abs(self.robot.get_joint_positions() - self.default_joint_pos)[control_idx]) < thresh

    @classmethod
    def create_for_data_processing(cls, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        return

    @classmethod
    def get_goal(self):
        raise NotImplementedError()

    @property
    def action_dimension(self):
        return self.robot.action_dim

    @property
    def default_joint_pos(self):
        if self.robot_model == "Tiago":
            default_q = np.array([
                -1.78029833e-04,  3.20231302e-05, -1.85759447e-07,
                0.0, -0.2,
                0.0,  0.1, -6.10000000e-01,
                -1.10000000e+00,  0.00000000e+00, -1.10000000e+00,  1.47000000e+00,
                0.00000000e+00,  8.70000000e-01,  2.71000000e+00,  1.50000000e+00,
                1.71000000e+00, -1.50000000e+00, -1.57000000e+00,  4.50000000e-01,
                1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
                4.50000000e-02,  4.50000000e-02,  4.50000000e-02
            ])
            if self.rigid_trunk:
                default_q[self.robot.trunk_control_idx] = self.robot.default_trunk_offset

        elif self.robot_model == "Fetch":
            default_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3, 1.23448, 1.8, -0.15, 1.36904, 1.90996, 0.05, 0.05])

        return default_q
    
    @property
    def env_args(self):
        # Includes all arguments needed to recreate this environment. Should be overwritten in child class.
        raise NotImplementedError()