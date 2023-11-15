import numpy as np

def get_control_limits(robot):
    """
    Get the control limits for the robot. This is used to normalize the action space.
    """
    
    control_limits = np.stack((-np.ones(robot.action_dim), np.ones(robot.action_dim)), axis=1)

    for name, cfg in robot._controller_config.items():
        joint_idx = cfg["dof_idx"]
        action_idx = robot.controller_action_idx[name]
        print(name)
        print(joint_idx)
        print(action_idx)
        # check if controller uses delta commands
        if "use_delta_commands" in cfg.keys():
            use_delta = cfg["use_delta_commands"]
        else:
            use_delta = False

        if cfg["name"] == "JointController":
            if len(joint_idx) == len(action_idx): # don't do anything to gripper action
                # print("hi")
                control_type = cfg["motor_type"]
                if use_delta:
                    control_limits[action_idx, 0] = np.stack(cfg["control_limits"][control_type], axis=1)[joint_idx, 0] - np.stack(cfg["control_limits"][control_type], axis=1)[joint_idx, 1]
                    control_limits[action_idx, 1] = -control_limits[action_idx, 0]
                else:
                    control_limits[action_idx, :] = np.stack(cfg["control_limits"][control_type], axis=1)[joint_idx, :]
        elif cfg["name"] == "InverseKinematicsController":
            assert cfg["motor_type"] == "velocity", "Controller must be in velocity mode"
            # assert cfg["mode"] == "pose_absolute_ori", "Controller must be in pose_delta_ori mode"
            if cfg["mode"] == "pose_absolute_ori":
                control_limits[action_idx, 0] = np.array([-0.055, -0.055, -0.055, -np.pi, -np.pi, -np.pi])
                control_limits[action_idx, 1] = np.array([0.055, 0.055, 0.055, np.pi, np.pi, np.pi])
            elif cfg["mode"] == "pose_delta_ori":
                control_limits[action_idx, 0] = np.array([-0.055, -0.055, -0.055, -np.pi, -np.pi, -np.pi])
                control_limits[action_idx, 1] = np.array([0.055, 0.055, 0.055, np.pi, np.pi, np.pi])
    return control_limits

def normalize_action(action, control_limits):
    """
    Normalize the action to [-1, 1] based on the control limits.
    """
    # return (action - control_limits[:, 0]) / (control_limits[:, 1] - control_limits[:, 0]) * 2 - 1
    eps = 1e-8
    return (action - control_limits[:, 0]) / (control_limits[:, 1] - control_limits[:, 0] + eps) * 2 - 1


def denormalize_action(action, control_limits):
    """
    Denormalize the action from [-1, 1] to the control limits.
    """
    eps = 1e-8
    return (action + 1) / 2 * (control_limits[:, 1] - control_limits[:, 0] + eps) + control_limits[:, 0]
