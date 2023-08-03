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

        if len(joint_idx) == len(action_idx):
            print("hi")
            control_type = cfg["motor_type"]
            control_limits[action_idx, :] = np.stack(cfg["control_limits"][control_type], axis=1)[joint_idx, :]
        
    return control_limits

def normalize_action(action, control_limits):
    """
    Normalize the action to [-1, 1] based on the control limits.
    """
    return (action - control_limits[:, 0]) / (control_limits[:, 1] - control_limits[:, 0]) * 2 - 1

def denormalize_action(action, control_limits):
    """
    Denormalize the action from [-1, 1] to the control limits.
    """
    return (action + 1) / 2 * (control_limits[:, 1] - control_limits[:, 0]) + control_limits[:, 0]