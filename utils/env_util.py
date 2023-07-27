from collections import defaultdict
import numpy as np
import json

def process_observation(obs, obs_modalities):
    step_obs_data = {}
    for mod in obs_modalities:
        if mod == "scan":
            mod_data = obs["robot0"]["robot0:laser_link_Lidar_sensor_scan"]
        elif mod == "rgb":
            mod_data = obs["robot0"]["robot0:eyes_Camera_sensor_rgb"]
            mod_data = mod_data[:, :, :3]
        elif mod == "depth":
            mod_data = obs["robot0"]["robot0:eyes_Camera_sensor_depth"]
            mod_data = mod_data[:, :, np.newaxis]
        elif mod == "proprio":
            mod_data = obs["robot0"]["proprio"]
        else:
            raise KeyError(f"{mod} is an invalid or unsupported modality for this robot.")
        step_obs_data[mod] = mod_data
    return step_obs_data

def get_skill_type(env, action, skill_type):
    base_action = action[env.robots[0].controller_action_idx["base"]]
    print("base action", base_action)
    if max(abs(base_action)) < 1e-8:
        return skill_type
    else:
        return skill_type + "_nav"

def process_traj_to_hdf5(traj_data, hdf5_file, traj_grp_name):
    data_grp = hdf5_file["data"]
    traj_grp = data_grp.create_group(traj_grp_name)
    traj_grp.attrs["num_samples"] = len(traj_data)
    
    obss = defaultdict(list)
    next_obss = defaultdict(list)
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
    next_obs_grp = traj_grp.create_group("next_obs")
    for mod, traj_mod_data in next_obss.items():
        next_obs_grp.create_dataset(mod, data=np.array(traj_mod_data))
    traj_grp.create_dataset("actions", data=np.array(actions))
    traj_grp.create_dataset("rewards", data=np.array(rewards))
    traj_grp.create_dataset("dones", data=np.array(dones))
