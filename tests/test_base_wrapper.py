import yaml
import numpy as np
import os
import sys
sys.path.append("../")
from envs.base_wrapper import OmnimimicBaseWrapper

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action.tolist())
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    env_wrapped = OmnimimicBaseWrapper(env, ["rgb", "joint_qpos", "eef_0_pos"], "data/test_base_wrapper.hdf5")
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    table = DatasetObject(
        name="table",
        category="breakfast_table",
        model="rjgmmy",
        scale = 0.3
    )
    og.sim.import_object(table)
    table.set_position([1.0, 1.0, 0.58])

    grasp_obj = DatasetObject(
        name="potato",
        category="cologne",
        model="lyipur",
        scale=0.01
    )

    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    controller = StarterSemanticActionPrimitives(None, scene, robot)

    # Need to set start pose because default tuck pose for Fetch collides with itself
    def set_start_pose():
        default_pose = np.array(
            [
                0.0,
                0.0,  # wheels
                0.0,  # trunk
                0.0,
                -1.0,
                0.0,  # head
                -1.0,
                1.53448,
                2.2,
                0.0,
                1.36904,
                1.90996,  # arm
                0.05,
                0.05,  # gripper
            ]
        )
        robot.set_joint_positions(default_pose)
        og.sim.step()

    def test_navigate_to_obj(env):
        set_start_pose()
        execute_controller(controller._navigate_to_obj(table), env)

    def test_grasp_no_navigation(env):
        # set_start_pose()
        robot.set_position([0.0, -0.5, 0.05])
        robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
        og.sim.step()
        execute_controller(controller.grasp(grasp_obj), env)

    def test_grasp(env):
        set_start_pose()
        execute_controller(controller.grasp(grasp_obj), env)

    robot.tuck()
    og.sim.step()
    test_grasp_no_navigation(env_wrapped)
    env_wrapped.save_data()
    print("Done")

if __name__ == "__main__":
    main()