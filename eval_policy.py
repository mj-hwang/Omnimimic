import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.file_utils import maybe_dict_from_checkpoint, config_from_checkpoint
from robomimic.algo import RolloutPolicy
from robomimic.algo import algo_factory
import numpy as np

from envs.nav_pick_env import NavPickEnv

ObsUtils.OBS_KEYS_TO_MODALITIES = {
    "proprio": "low_dim",
    "rgb" : "rgb",
}

# load checkpoint
# ckpt_dict = maybe_dict_from_checkpoint(ckpt_path="/home/svl/robomimic/robomimic/output_dir2520/bc_rnn_example/20230726131451/models/model_epoch_100.pth")
ckpt_dict = maybe_dict_from_checkpoint(ckpt_path="/scr/garlanka/robomimic/robomimic/testingg/bc_rnn_example/20230726144913/models/model_epoch_50.pth")
algo_name = ckpt_dict["algo_name"]
config, _ = config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict)

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

# breakpoint()

env = NavPickEnv(
    max_steps=1000,
    random_init_obj=True, # randomly initialize object position on table
    # random_init_base_pose=[[-1.0, 0.0], [1.0, 0.5], [-np.pi, np.pi]], # randomly initialize robot base pose
    # random_init_robot_pose=False, # randomly initialize robot base and arm poses - TODO
    initial_robot_pose=[-0.8, 0.5, 0.0], # defualt initial pose [x, y, z, yaw] 
    raise_hand=True,
    fix_arm_during_nav=False,
)
obs = env.reset()
obs = {
    "proprio" : obs["robot0"]["proprio"],
    "rgb" : np.moveaxis(obs["robot0"]["robot0:eyes_Camera_sensor_rgb"], -1, 0)[:3, :, :],
}

horizon = 400
total_return = 0
for step_i in range(horizon):
    # get action from policy (calls @get_action)
    act = policy(obs)
    # play action
    next_obs, r, done, _ = env.step(act)
    obs = {
        "proprio" : next_obs["robot0"]["proprio"],
        "rgb" : np.moveaxis(next_obs["robot0"]["robot0:eyes_Camera_sensor_rgb"], -1, 0)[:3, :, :],
    }

    # total_return += r
    # success = env.is_success()["task"]
    # if done or success:
    #     break