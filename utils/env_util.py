from collections import defaultdict
import numpy as np

def process_observation(env, obs, obs_modalities):
    proprioception_dict = env.robots[0]._get_proprioception_dict()

    step_obs_data = {}
    for mod in obs_modalities:
        if mod == "lidar":
            mod_data = obs["robot0"]["robot0:laser_link_Lidar_sensor_scan"]
        elif mod == "rgb":
            mod_data = obs["robot0"]["robot0:eyes_Camera_sensor_rgb"]
        elif mod == "depth":
            mod_data = obs["robot0"]["robot0:eyes_Camera_sensor_depth"]
        elif mod == "robot_pos":
            mod_data = env.robots[0].get_position()
        elif mod == "robot_quat":
            mod_data = env.robots[0].get_rpy()
        elif mod in proprioception_dict:
            mod_data = proprioception_dict[mod]
        else:
            raise KeyError(f"{mod} is an invalid or unsupported modality for this robot.")
        step_obs_data[mod] = mod_data
    return step_obs_data

# def process_step(env, obs_data, action, next_obs, reward, done, obs_modalities):
#     """
#     Parse OmniGibson config file / object

#     Args:
#         config (dict or str): Either config dictionary or path to yaml config to load

#     Returns:
#         dict: Parsed config
#     """
#     step_data = {}

#     step_data["obs"] = obs
#     step_data["action"] = action
#     step_data["reward"] = reward
#     step_data["next_obs"] = process_observation(next_obs)
#     step_data["done"] = True

#     return step_data

def process_traj_to_hdf5(traj_data, hdf5_file, traj_count, skill_type):
    data_grp = hdf5_file["data"]
    traj_grp = data_grp.create_group(f"demo_{traj_count}")
    traj_grp.attrs["num_samples"] = len(traj_data)
    
    obss = defaultdict([])
    next_obss = defaultdict([])
    actions = []
    rewards = []
    dones = []

    for step_data in traj_data:
        for mod, step_mod_data in step_data["obs"].items():
            obss[mod].append(step_mod_data)
        for mod, step_mod_data in step_data["next_obs"].items():
            next_obss[mod].append(step_mod_data)
        actions.append(step_data["action"])
        rewards.append(step_data["reward"])
        dones.append(step_data["done"])

    obs_grp = traj_grp.create_group("obs")
    for mod, traj_mod_data in obss.items():
        obs_grp.create_dataset(mod, data=np.array(traj_mod_data))
    next_obs_grp = traj_grp.create_group("obs")
    for mod, traj_mod_data in next_obss.items():
        next_obs_grp.create_dataset(mod, data=np.array(traj_mod_data))
    traj_grp.create_dataset("action", data=np.array(actions))
    traj_grp.create_dataset("rewards", data=np.array(rewards))
    traj_grp.create_dataset("dones", data=np.array(dones))
    
    # TODO: add a mask for skill type
    # if skill_type is not None:
    #     hdf5_file["mask"]
