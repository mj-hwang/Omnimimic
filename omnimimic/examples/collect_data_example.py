from omnimimic.envs.skill_wrapper import OmnimimicSkillWrapper

from example_custom_envs.nav_pick_env import NavPickEnv

import time
import numpy as np
import argparse

def collect_data(args):
    
    # initialize environment
    moma_env_wrapper = NavPickEnv(
        max_steps=700,
        config_filename="nav_pick.yaml",
        random_init_obj=True, # randomly initialize object position on table
        random_init_base_pose=[[-0.65, -0.55], [-0.3, -0.2], [-0.8*np.pi, -0.7*np.pi]], # randomly initialize robot base pose
        raise_hand=True,
        head_mode="auto",
        simple_scene=True,
    )

    # wrap it in skill wrapper
    env_wrapped = OmnimimicSkillWrapper(
        moma_env_wrapper, 
        ["scan", "rgb", "depth", "rgb_wrist", "depth_wrist", "proprio"], 
        f"{args.data_path}",
    )

    controller = moma_env_wrapper.controller
    done_collecting = False
    start_time = time.time()
    while not done_collecting: 
        env_wrapped.execute_skill(controller._grasp(moma_env_wrapper.objects["grasp_obj"]), "pick")
        env_wrapped.reset()
        time.sleep(1)

        done_collecting = env_wrapped.manipulation_traj_count >= args.n_traj
        print(f"collected {env_wrapped.manipulation_traj_count} trajectories")
        print("time elapsed:", time.time() - start_time)
    print("Done")

def main(args):
    collect_data(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-d", type=str, default="test.hdf5")
    parser.add_argument("--n_traj", "-n", type=int, default=10)
    args = parser.parse_args()
    main(args)