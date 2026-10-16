import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)
from robosuite.utils.transform_utils import quat2axisangle

from place_box_in_bin_env import PlaceBoxInBinEnv, TABLE_OFFSET


import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def clear_coordinate_systems(env):
    viewer = env.viewer
    if viewer is None:
        return

    scene = viewer.viewer.user_scn
    scene.ngeom = 0  # Clear previously drawn geometries


def draw_coordinate_system(env, position, quaternion, axis_length=0.2):
    viewer = env.viewer
    if viewer is None:
        return

    scene = viewer.viewer.user_scn

    position = np.asarray(position, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float)
    rotation = Rotation.from_quat(quaternion).as_matrix()

    # Local X, Y, Z axes
    local_axes = [
        (np.array([1, 0, 0]), [1, 0, 0, 1]),  # X: red
        (np.array([0, 1, 0]), [0, 1, 0, 1]),  # Y: green
        (np.array([0, 0, 1]), [0, 0, 1, 1]),  # Z: blue
    ]

    for direction, color in local_axes:
        endpoint = position + rotation @ (direction * axis_length)

        geom = scene.geoms[scene.ngeom]

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array(color, dtype=np.float32),
        )

        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.005,
            position,
            endpoint,
        )

        scene.ngeom += 1


def main():
    # Load controller configuration
    arm_config = suite.load_part_controller_config(
        default_controller="OSC_POSE"
    )
    arm_config["input_type"] = "absolute"
    arm_config["input_ref_frame"] = "world"
    controller_config = refactor_composite_controller_config(
        arm_config,
        "Panda",
        ["right"],
    )

    # Create simulation environment
    env = suite.make(
        env_name="PlaceBoxInBinEnv",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,
        control_freq=20,
    )

    obs = env.reset()
    ee_pos = obs["robot0_eef_pos"]
    ee_quat = obs["robot0_eef_quat_site"]
    hold_action = np.concatenate([ee_pos, quat2axisangle(ee_quat), [-1.0]])

    for _ in range(10000):
        start = time.time()
        obs, _, _, _ = env.step(hold_action)

        clear_coordinate_systems(env)

        # Draw axes at table "origin"
        draw_coordinate_system(env, TABLE_OFFSET, (0.0, 0.0, 0.0, 1.0), axis_length=0.2)

        # Draw axes at current end-effector pose
        draw_coordinate_system(env, ee_pos, ee_quat, axis_length=0.2)

        # Limit frame rate
        elapsed = time.time() - start
        diff = 1 / 20 - elapsed
        if diff > 0:
            time.sleep(diff)


if __name__ == "__main__":
    main()
