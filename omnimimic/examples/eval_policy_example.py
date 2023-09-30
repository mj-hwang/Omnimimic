import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import maybe_dict_from_checkpoint, config_from_checkpoint
from robomimic.algo import RolloutPolicy
from robomimic.algo import algo_factory
import numpy as np
import argparse

from example_custom_envs.nav_pick_env import NavPickEnv
from omnimimic.utils.robot_util import *

KEYS_TO_MODALITIES = {
    "proprio": "low_dim",
    "rgb" : "rgb",
    "depth" : "depth",
    "scan" : "scan",
    "rgb_wrist" : "rgb",
    "depth_wrist" : "depth",
}

def main(args):

    # load checkpoint
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=args.model)

    algo_name = ckpt_dict["algo_name"]
    config, _ = config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict)

    # define observations to use
    ObsUtils.OBS_KEYS_TO_MODALITIES = {}
    # if no obs keys are specified, use everything
    if args.obs is None:
        obs_to_use = list(KEYS_TO_MODALITIES.keys())
    else:
        obs_to_use = args.obs
    for obs_key in obs_to_use:
        ObsUtils.OBS_KEYS_TO_MODALITIES[obs_key] = KEYS_TO_MODALITIES[obs_key]        

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # create Algo instance
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=ckpt_dict["shape_metadata"]["all_shapes"],
        ac_dim=ckpt_dict["shape_metadata"]["ac_dim"],
        device=device,
    )

    # load weights
    model.deserialize(ckpt_dict["model"])
    model.set_eval()

    # rollout wrapper around model
    policy = RolloutPolicy(model)

    # @policy should be instance of RolloutPolicy
    assert isinstance(policy, RolloutPolicy)

    # episode reset (calls @set_eval and @reset)
    policy.start_episode()

    # load environment from model checkpoint's env args
    env_kwargs = ckpt_dict["env_metadata"]["env_kwargs"]
    print("loading env from ckpt with kwargs: \n", env_kwargs)
    env = NavPickEnv(**env_kwargs)
    
    obs = env.reset()

    control_limits = get_control_limits(env.robot)

    n_success = 0
    n_rollouts = args.n
    for i in range(n_rollouts):
        print("evaluation rollout ", i)
        done = False
        obs = env.reset()
        obs = populate_obs(obs, obs_to_use)
        step = 0        
        while not done:
            step += 1
            print(step)
            # get action from policy (calls @get_action)
            act = policy(obs)
            act = denormalize_action(act, control_limits)

            act[env.robot.controller_action_idx["arm_right"]] = 0.0
            act[env.robot.controller_action_idx["base"]] = 0.0

            next_obs, reward, done, info = env.step(act)
            
            # process observations
            obs = populate_obs(next_obs, obs_to_use)

            if env.is_success():
                n_success += 1  

        print(f"episode {i} done. success = {env.is_success()}") 

    print(f"{n_success} out of {n_rollouts} rollouts succeeded")
    print("success rate: ", n_success / n_rollouts)

def populate_obs(observations, modality_keys):
    obs = {}
    for key in modality_keys:
        if key == "proprio":
            obs["proprio"] = observations["robot0"]["proprio"]
        elif key == "rgb":
            data = observations["robot0"]["robot0:eyes_Camera_sensor_rgb"][:, :, :3]
            obs["rgb"] = ObsUtils.process_obs(data, obs_modality="rgb")
        elif key == "depth":
            data = observations["robot0"]["robot0:eyes_Camera_sensor_depth"][:, :, np.newaxis]
            obs["depth"] = ObsUtils.process_obs(data, obs_modality="depth")
        elif key == "rgb_wrist":
            data = observations["robot0"]["robot0:wrist_camera_frame_Camera_sensor_rgb"][:, :, :3]
            obs["rgb_wrist"] = ObsUtils.process_obs(data, obs_modality="rgb")
        elif key == "depth_wrist":
            data = observations["robot0"]["robot0:wrist_camera_frame_Camera_sensor_depth"][:, :, np.newaxis]
            obs["depth_wrist"] = ObsUtils.process_obs(data, obs_modality="depth")
        elif key == "scan":
            obs["scan"] = np.concatenate([
                                observations["robot0"]["robot0:base_front_laser_link_Lidar_sensor_scan"],
                                observations["robot0"]["robot0:base_rear_laser_link_Lidar_sensor_scan"]
                            ]).T
        else:
            raise Exception(f"invalid key {key}")
    return obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--obs", "-o", default=None, nargs='*', type=str, help="observations to use: e.g. --obs proprio rgb rgb_wrist scan")
    parser.add_argument("--n", type=int, default=10, help="number of evaluation rollouts")

    args = parser.parse_args()
    main(args)
    