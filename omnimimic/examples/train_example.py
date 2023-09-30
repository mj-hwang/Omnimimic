import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

from omnimimic.utils.macros import gm as omnimimic_gm

import argparse
import json
import os

def main(args):

    # load config file
    config_path = os.path.join(omnimimic_gm.ROOT_DIR, "examples/example_custom_envs/configs/example_train.json")
    data_path = args.data_path
    print(f"config path: {config_path}")
    ext_cfg = json.load(open(config_path, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
        config.train.data = data_path
        config.train.output_dir = args.out_dir

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # launch training run
    train(config, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test script")

    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        required=True,
        help="path to dataset",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="test",
    )

    args = parser.parse_args()

    main(args)

