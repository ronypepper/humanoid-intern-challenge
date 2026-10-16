import argparse
from pathlib import Path
import numpy as np
import h5py
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

from demonstration_sampler import DemonstrationSampler
from place_box_in_bin_env import PlaceBoxInBinEnv, TABLE_OFFSET


GRIPPER_OPEN_THRESHOLD = 0.085
GRIPPER_CLOSE_THRESHOLD = 0.09


def play_and_convert_dataset(demos_dataset_path: Path, sim_dataset_path: Path, control_frequency: float = 20):
    # Load datasets
    demos_dataset_file = h5py.File(demos_dataset_path, "r")
    sim_dataset_file = h5py.File(sim_dataset_path, "w")

    # Object reset handler
    objects_initializer = DemonstrationSampler("demo_sampler", 
                                               demos_dataset_file, 
                                               table_offset=TABLE_OFFSET,
                                               z_offset=0.01)

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
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_eye_in_hand", "frontview"],
        camera_heights=224,
        camera_widths=224,
        control_freq=control_frequency,
        placement_initializer=objects_initializer
    )

    # Demonstrations rollout loop
    for demo_idx in range(len(demos_dataset_file)):
        print(f"[INFO] Starting rollout {demo_idx + 1}/{len(demos_dataset_file)}")

        # Prepare datasets for next demonstration
        demo_dataset = demos_dataset_file[f"demo_{demo_idx}"]
        sim_dataset = sim_dataset_file.create_group(f"demo_{demo_idx}")
        sim_dataset.attrs["prompt"] = demo_dataset.attrs["prompt"]
        sim_dataset.attrs["control_frequency"] = control_frequency
        joint_obss, joint_acts, wrist_imgs, table_imgs = [], [], [], []
        num_samples = len(demo_dataset["gripper/timing"])

        # Play one demonstration
        demo_time = demo_dataset["gripper/timing"][0]
        cur_sample_idx = 0
        action = None
        gripper_cmd = -1
        obs = env.reset()
        while True:
            # Determine next sample point from demonstration dataset (or if end has been reached)
            end_of_dataset_reached = False
            while True:
                if num_samples <= cur_sample_idx + 1:
                    end_of_dataset_reached = True
                    break
                if demo_dataset["gripper/timing"][cur_sample_idx + 1] > demo_time:
                    break
                cur_sample_idx += 1
                action = None  # Must process new action
            if end_of_dataset_reached:
                break

            # Sample next action from demonstration dataset
            if action is None:
                gripper_T = demo_dataset["gripper/transform"][cur_sample_idx]
                gripper_pos = gripper_T[:3, 3] + TABLE_OFFSET
                gripper_rot = Rotation.from_matrix(gripper_T[:3, :3]).as_rotvec()
                gripper_opening = demo_dataset["gripper/opening"][[cur_sample_idx]]
                if gripper_opening > GRIPPER_CLOSE_THRESHOLD:
                    gripper_cmd = 1.0
                elif gripper_opening < GRIPPER_OPEN_THRESHOLD:
                    gripper_cmd = -1.0
                action = np.concat((gripper_pos, gripper_rot, [gripper_cmd]))

            # Apply action to simulation
            obs, _, _, _ = env.step(action)
            demo_time += 1.0 / control_frequency

            # Process and store observations and actions for dataset
            joint_obs = np.concat((obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][[0]]), axis=0)
            joint_act = np.copy(action)
            wrist_img = obs["robot0_eye_in_hand_image"][::-1, :, ::-1]
            table_img = obs["frontview_image"][::-1, :, ::-1]

            joint_obss.append(joint_obs)
            joint_acts.append(joint_act)
            wrist_imgs.append(wrist_img)
            table_imgs.append(table_img)

        # Save observations and actions to dataset
        sim_dataset["obs/joint_pos"] = joint_obss
        sim_dataset["act/joint_pos"] = joint_acts
        sim_dataset["obs/wrist_img"] = wrist_imgs
        sim_dataset["obs/table_img"] = table_imgs

    print("[INFO] Done with replay.")

    # Cleanup
    demos_dataset_file.close()
    sim_dataset_file.close()
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert demonstrations collected with record_demonstrations.py into a VLA-trainable"
                    "format by playing it in a matching RoboSuite simulation environment."
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to dataset created by record_demonstrations.py")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path where converted dataset will be saved")
    parser.add_argument("-c", "--control_frequency", type=float, required=True, help="Control frequency of RoboSuite simulation")

    return parser.parse_args()


def main():
    args = parse_args()
    play_and_convert_dataset(args.input, args.output, args.control_frequency)


if __name__ == "__main__":
    main()
