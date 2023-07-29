
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.file_utils import maybe_dict_from_checkpoint, config_from_checkpoint
from robomimic.algo import RolloutPolicy
from robomimic.algo import algo_factory

sys.path.append("../")
from envs.eval_wrapper import OmnimimicEvalWrapper

def eval_checkpoint(ckpt_path, omni_env, num_frames=1):
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    algo_name = ckpt_dict["algo_name"]
    config, _ = config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=ckpt_dict["shape_metadata"]["all_shapes"],
        ac_dim=ckpt_dict["shape_metadata"]["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    model.set_eval()

    # rollout wrapper around model
    policy = RolloutPolicy(model)
    policy.start_episode()

    eval_env = OmnimimicEvalWrapper(omni_env, num_frames=num_frames)
    obs = eval_env.reset()

    horizon = 400
    for _ in range(horizon):
        act = policy(obs)
        obs, r, done, _ = eval_env.step(act)

        # TODO: add eval statistics
        if done:
            break
