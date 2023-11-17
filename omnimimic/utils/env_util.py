from collections import defaultdict
import numpy as np

from robomimic.utils.obs_utils import process_obs

# from PIL import Image
# import time

def process_omni_obs(obs, obs_modalities, postprocess_for_eval=False):
    step_obs_data = {}
    for mod in obs_modalities:
        if mod == "scan":
            if "robot0:laser_link_Lidar_sensor_scan" in obs["robot0"]:
                mod_data = obs["robot0"]["robot0:laser_link_Lidar_sensor_scan"]
            else:
                mod_data = np.concatenate([
                    obs["robot0"]["robot0:base_front_laser_link_Lidar_sensor_scan"],
                    obs["robot0"]["robot0:base_rear_laser_link_Lidar_sensor_scan"]
                ])
            mod_data = mod_data.T.copy()
        elif mod == "rgb":
            mod_data = obs["robot0"]["robot0:eyes_Camera_sensor_rgb"]
            mod_data = mod_data[:, :, :3].copy()
            if postprocess_for_eval:
                mod_data = process_obs(mod_data, obs_modality="rgb")      
        elif mod == "depth":
            mod_data = obs["robot0"]["robot0:eyes_Camera_sensor_depth"]
            mod_data = mod_data[:, :, np.newaxis].copy()
            if postprocess_for_eval:
                mod_data = process_obs(mod_data, obs_modality="depth") 
        elif mod == "rgb_wrist":
            mod_data = obs["robot0"]["robot0:wrist_camera_frame_Camera_sensor_rgb"]
            mod_data = mod_data[:, :, :3].copy()
            if postprocess_for_eval:
                mod_data = process_obs(mod_data, obs_modality="rgb")
        elif mod == "depth_wrist":
            mod_data = obs["robot0"]["robot0:wrist_camera_frame_Camera_sensor_depth"]
            mod_data = mod_data[:, :, np.newaxis].copy()
            if postprocess_for_eval:
                mod_data = process_obs(mod_data, obs_modality="depth")
        elif mod == "rgb_external":
            mod_data = obs["external"]["rgb"]
            mod_data = mod_data[:, :, :3].copy()
            # debug_img = Image.fromarray(mod_data)
            # debug_img.save(f"debug/debug_{time.time()}.png")
            if postprocess_for_eval:
                mod_data = process_obs(mod_data, obs_modality="rgb")
        elif mod == "depth_external":
            mod_data = obs["external"]["depth"].copy()
            mod_data = mod_data[:, :, np.newaxis]
            if postprocess_for_eval:
                mod_data = process_obs(mod_data, obs_modality="depth")
        elif mod == "proprio":
            mod_data = obs["robot0"]["proprio"].copy()
        else:
            raise KeyError(f"{mod} is an invalid or unsupported modality for this robot.")
        step_obs_data[mod] = mod_data
    return step_obs_data

def get_skill_type(env, action, skill_type):
    base_action = action[env.robots[0].controller_action_idx["base"]]
    # print("base action", base_action)
    if max(abs(base_action)) < 1e-8:
        return skill_type + "_manip"
    else:
        return skill_type + "_nav"

def process_traj_to_hdf5(traj_data, hdf5_file, traj_grp_name, save_next_obs=False):
    data_grp = hdf5_file.require_group("data")
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
        if save_next_obs:
            for mod, step_mod_data in step_data["next_obs"].items():
                next_obss[mod].append(step_mod_data)
        actions.append(step_data["action"])
        rewards.append(step_data["reward"])
        dones.append(step_data["done"])

    # np.save(f"IK_test/hdf5_action_{time.time()}.npy", np.stack(actions, axis=0))

    obs_grp = traj_grp.create_group("obs")
    for mod, traj_mod_data in obss.items():
        obs_grp.create_dataset(mod, data=np.stack(traj_mod_data, axis=0))
    if save_next_obs:
        next_obs_grp = traj_grp.create_group("next_obs")
        for mod, traj_mod_data in next_obss.items():
            next_obs_grp.create_dataset(mod, data=np.stack(traj_mod_data, axis=0))
    traj_grp.create_dataset("actions", data=np.stack(actions, axis=0))
    traj_grp.create_dataset("rewards", data=np.stack(rewards, axis=0))
    traj_grp.create_dataset("dones", data=np.stack(dones, axis=0))
