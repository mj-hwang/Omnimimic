import omnigibson as og
from omnigibson.objects.dataset_object import DatasetObject
import omnigibson.utils.transform_utils as T

from .moma_env import MOMAEnv

import numpy as np

class NavPickEnv(MOMAEnv):
    """
    Example environment for distilling-moma project.
    A Tiago robot is tasked to navigate to a table and pick up a bottle of cologne from the table.

    Args:
        random_init_obj (bool): randomly initialize object position on table
        random_init_base_pose (list): initial robot base pose is uniformly sampled from the specified range
            expects [[x_min, x_max], [y_min, y_max], [yaw_min, yaw_max]]
        random_init_arm_pose (dict): whether to apply random noise to arm joint positions at initialization
            expects { arm1 : bool, arm2 : bool } (e.g. { "left" : True, "right" : False })
        initial_robot_pose (list): initial robot base pose [x, y, z, yaw]. Is ignored if @random_init_base_pose is set
        raise_hand (bool): if True, sets the defalt arm reset position to above the head. If False, use default tucked position
        rigid_trunk (bool): if True, sets the trunk to be rigid
        head_mode (str): None, "auto" or "fixed". None does not overwrite head action, "auto" tracks object to grasp, "fixed" keeps head stationary
        localization_noise (dict): mean and std of noise to add to robot localization # NOTE: not implemented yet
        simple_scene (bool): if True, only load floor, wall and table
    """
    
    def __init__(
            self,
            env_name="NavPickEnv",
            max_steps=1000,
            config_filename="nav_pick.yaml",
            random_init_obj=False, # randomly initialize object position on table
            random_init_base_pose=None, # randomly initialize robot base pose
            random_init_arm_pose=None, 
            initial_robot_pose=[0.0, 0.0, 0.0], # defualt initial pose [x, y, z, yaw] 
            raise_hand=True,
            rigid_trunk=False,
            head_mode=None, # None, "auto" or "fixed". None does not overwrite head action, "auto" tracks object to grasp, "fixed" keeps head stationary
            localization_noise={
                "mean" : (0.0, 0.0, 0.0),
                "std" : (0.0, 0.0, 0.0),
            },
            simple_scene=False,
    ):

        super().__init__(
            env_name=env_name,
            config_filename=config_filename,
            max_steps=max_steps,
            random_init_base_pose=random_init_base_pose,
            random_init_arm_pose=random_init_arm_pose,
            initial_robot_pose=initial_robot_pose,
            raise_hand=raise_hand,
            rigid_trunk=rigid_trunk,
            localization_noise=localization_noise,
            simple_scene=simple_scene,
        )

        self.random_init_obj = random_init_obj
        self.grasp_obj_default_pos = [-0.47616568, -1.21954441, 0.5]
        self.head_mode = head_mode
        self.fixed_head_pos = np.array([0.3, -1.0])
        self.simple_scene = simple_scene
        
        # set head joints
        if self.head_mode == "fixed":
            self._set_head_joint()

        self.add_objects()
        self._place_objects()

        self.step_sim(10)

    def add_objects(self):

        grasp_obj = DatasetObject(
            name="grasp_obj",
            category="bottle_of_cologne", # for new asset version
            model="lyipur",
            scale=[1.0, 1.0, 1.0], # for new asset version
        )
        og.sim.import_object(grasp_obj)

        # update parent class objects dict
        self.add_to_objects_dict([grasp_obj])

    def _place_objects(self):

        pos = self._sample_grasp_obj_pos()
        quat = [0,0,1,0]
        obj = self.objects["grasp_obj"]
        obj.set_position_orientation(pos, quat)
        self.step_sim(10)
        obj.set_orientation(quat)
        self.step_sim(10)
        print("Placed object at: ", pos)

    def _sample_grasp_obj_pos(self):
        """
        Sample a pose for grasp_obj
        """
        if not self.random_init_obj:
            # return self.grasp_obj_default_pos
            return np.array([-0.3, -0.8, 0.5])
        
        # delta_pos = np.random.uniform(low=-0.1, high=0.1, size=2)
        delta_pos = np.random.uniform(low=-0.03, high=0.03, size=2)
        pos = np.array([self.grasp_obj_default_pos[0] + delta_pos[0], self.grasp_obj_default_pos[1] + 0.45 + delta_pos[1], self.grasp_obj_default_pos[2]])
        return pos

    def check_termination(self):
        """
        Check if termination condition has been met. Overwrite this in child class.

        Returns:
            done (bool): True if termination condition has been met
            info (dict): dictionary of termination conditions
        """
        success = self.is_success()
        failure, failure_info = self._check_failure()
        done = success or failure
        info = {"success": success, "failure": failure}
        info.update(failure_info)
        return done, info

    def is_success(self):
        """
        Success confition: robot picked up the object
        """
        # print("-------potato check success---------")

        grasping_obj = self._is_grasping_obj("grasp_obj")
        obj_lifted = self.objects["grasp_obj"].get_position()[2] > 0.45
        
        successes = grasping_obj and obj_lifted
        return successes

    def _check_failure(self): 
        """
        Failure conditions:
        - object is on the floor
        - max number of steps reached
        """
        # print("-------potato check failure---------")

        horizon_reached = self._check_horizon_reached()
        dropped_obj = self._is_obj_on_floor(self.objects["grasp_obj"])
        failed = dropped_obj or horizon_reached
        failed_info = {
            "dropped_grasp_obj" : dropped_obj,
            "horizon_reached" : horizon_reached,
        }
        return failed, failed_info

    def step(self, action):
        if self.head_mode is not None:
            action = self.overwrite_head_action(action, self.objects["grasp_obj"], mode=self.head_mode)
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info
    
    def reset(self):
        print("Resetting NavPickEnv...")
        # reset env place objects
        # set gripper joints to open position, set head position -> take a few sim steps
        self.open_grippers()
        obs = super().reset()
        self._set_head_joint()
        self._place_objects()
        self.step_sim(20)
        obs["robot0"] = self.robot.get_obs()
        return obs
    
    def overwrite_head_action(self, action, obj, mode):
        assert self.robot_model == "Tiago", "Tracking object with camera is currently only supported for Tiago"
        head_idx = self.robot.controller_action_idx["camera"]
        
        # check controller type and mode
        config = self.robot._controller_config["camera"]
        assert config["name"] == "JointController", "Camera controller must be JointController"
        assert config["motor_type"] == "position", "Camera controller must be in position control mode"
        use_delta = config["use_delta_commands"]
        
        if mode == "auto":
            head_q = self.get_head_goal_q(obj)
            
            if use_delta:
                cur_head_q = self.robot.get_joint_positions()[self.robot.camera_control_idx]
                head_action = head_q - cur_head_q
            else:
                head_action = head_q
            action[head_idx] = head_action
            return action

        elif mode == "fixed":
            if use_delta:
                return action
            else:
                action[head_idx] = self.fixed_head_pos
                return action

    def get_head_goal_q(self, obj):
        """
        Get goal joint positions for head to look at an object of interest,
        If the object cannot be seen, return the current head joint positions.
        """

        # get current head joint positions
        head1_joint = self.robot.joints["head_1_joint"]
        head2_joint = self.robot.joints["head_2_joint"]
        head1_joint_limits = [head1_joint.lower_limit, head1_joint.upper_limit]
        head2_joint_limits = [head2_joint.lower_limit, head2_joint.upper_limit]
        head1_joint_goal = head1_joint.get_state()[0][0]
        head2_joint_goal = head2_joint.get_state()[0][0]

        # grab robot and object poses
        robot_pose = self.robot.get_position_orientation()
        obj_pose = obj.get_position_orientation()
        obj_in_base = T.relative_pose_transform(*obj_pose, *robot_pose)

        # compute angle between base and object in xy plane (parallel to floor)
        theta = np.arctan2(obj_in_base[0][1], obj_in_base[0][0])
        
        # if it is possible to get object in view, compute both head joint positions
        if head1_joint_limits[0] < theta < head1_joint_limits[1]:
            head1_joint_goal = theta
            
            # compute angle between base and object in xz plane (perpendicular to floor)
            head2_pose = self.robot.links["head_2_link"].get_position_orientation()
            head2_in_base = T.relative_pose_transform(*head2_pose, *robot_pose)

            phi = np.arctan2(obj_in_base[0][2] - head2_in_base[0][2], obj_in_base[0][0])
            if head2_joint_limits[0] < phi < head2_joint_limits[1]:
                head2_joint_goal = phi

        # if not possible to look at object, return default head joint positions
        else:
            default_head_pos = self.default_joint_pos[self.robot.controller_action_idx["camera"]]
            head1_joint_goal = default_head_pos[0]
            head2_joint_goal = default_head_pos[1]

        return [head1_joint_goal, head2_joint_goal]
    
    def _set_head_joint(self):
        self.robot.set_joint_positions(self.fixed_head_pos, self.robot.camera_control_idx)
        self.step_sim(10)

    @property
    def env_args(self):
        env_args = {
            "env_name" : self.name,
            "type" : 4,
            "env_kwargs" : {
                "max_steps" : self.max_steps,
                "random_init_obj" : self.random_init_obj,
                "random_init_base_pose" : self.random_init_base_pose,
                "random_init_arm_pose" : self.random_init_arm_pose,
                "initial_robot_pose" : self.initial_robot_pose, 
                "raise_hand" : self.raise_hand,
                "rigid_trunk" : self.rigid_trunk,
                "head_mode" : self.head_mode,
                "simple_scene" : self.simple_scene,
            },
        }
        return env_args