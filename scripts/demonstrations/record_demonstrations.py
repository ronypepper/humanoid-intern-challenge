import argparse
import tomllib
from pathlib import Path
import os
import cv2 as cv

from charuco_board_tracker import CharucoBoardConfig
from demo_recorder import DemoRecorderConfig, DemoRecorder, CameraConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record robotic demonstration data via hand and charuco-marker based object tracking."
    )
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to the TOML configuration file",)
    parser.add_argument("-d", "--dataset", type=Path, required=True, help="Path where demonstrations will be stored in a hdf5 dataset")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt corresponding to the demonstrations")
    parser.add_argument("-n", "--num", type=int, required=True, help="Number of demonstrations to record")
    parser.add_argument("-o", "--overwrite", action='store_true', help="Overwrite an existing dataset")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check dataset path
    if args.dataset.exists():
        if args.overwrite:
            os.remove(args.dataset)
        else:
            raise RuntimeError("[ERROR] Dataset file already exists! Use --overwrite flag to overwrite it.")

    # Load configuration from toml file
    if not args.config.exists():
        raise ValueError(f"Config file {args.config} does not exist!")
    with args.config.open("rb") as f:
        file_config = tomllib.load(f)
        cam_hands_a_cfg = CameraConfig(file_config["camera_hands_a"]["device_id"],
                                       file_config["camera_hands_a"]["width"],
                                       file_config["camera_hands_a"]["height"],
                                       file_config["camera_hands_a"]["calibration_path"])
        cam_hands_b_cfg = CameraConfig(file_config["camera_hands_b"]["device_id"],
                                       file_config["camera_hands_b"]["width"],
                                       file_config["camera_hands_b"]["height"],
                                       file_config["camera_hands_b"]["calibration_path"])
        cam_objects_cfg = CameraConfig(file_config["camera_hands_b"]["device_id"],
                                       file_config["camera_hands_b"]["width"],
                                       file_config["camera_hands_b"]["height"],
                                       file_config["camera_hands_b"]["calibration_path"])
        origin_cfg = CharucoBoardConfig((file_config["origin"]["charuco"]["columns"], 
                                         file_config["origin"]["charuco"]["rows"]),
                                        file_config["origin"]["charuco"]["square_size"],
                                        file_config["origin"]["charuco"]["marker_size"],
                                        file_config["origin"]["charuco"]["aruco_dict"],
                                        file_config["origin"]["charuco"]["aruco_dict_offset"])
        world_origin_offset = tuple(file_config["origin"]["origin_offset"])
        cfg = DemoRecorderConfig(cam_hands_a_cfg, 
                                 cam_hands_b_cfg,
                                 cam_objects_cfg,
                                 origin_cfg,
                                 world_origin_offset,
                                 file_config["general"]["gripper_hand"],
                                 file_config["general"]["handlandmarker_model_asset_path"],
                                 file_config["general"]["visualize_hands"])

        objects = {}
        obj_columns = file_config["objects"]["charuco"]["columns"]
        obj_rows = file_config["objects"]["charuco"]["rows"]
        obj_square_size = file_config["objects"]["charuco"]["square_size"]
        obj_marker_size = file_config["objects"]["charuco"]["marker_size"]
        obj_aruco_dict = file_config["objects"]["charuco"]["aruco_dict"]
        for obj in file_config["object"]:
            charuco_cfg = CharucoBoardConfig((obj_columns, obj_rows), 
                                             obj_square_size, 
                                             obj_marker_size, 
                                             obj_aruco_dict,
                                             obj["aruco_dict_offset"])
            origin_offset = tuple(obj["origin_offset"])
            objects[obj["name"]] = (charuco_cfg, origin_offset)

    # Initialize demonstration recorder
    print(f"[INFO] Initializing demonstration recorder...")
    demo_recorder = DemoRecorder(cfg)
    for name, (charuco_cfg, origin_offset) in objects.items():
        demo_recorder.add_tracked_object(name, charuco_cfg, origin_offset)

    # Record demonstrations
    for i in range(args.num):
        print(f"[INFO] Ready to record demonstration {i+1}/{args.num}...")
        demo_recorder.record_demonstration(args.prompt, args.dataset)
        print(f"[INFO] Demonstration {i+1}/{args.num} recorded.")


if __name__ == "__main__":
    cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
    main()
