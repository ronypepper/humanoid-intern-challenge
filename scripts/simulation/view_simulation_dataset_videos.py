import argparse
from pathlib import Path
import time
import h5py
import cv2 as cv
import numpy as np


def playback_dataset_videos(dataset_path: Path, loop: bool):
    # Load dataset
    dataset_file = h5py.File(dataset_path, "r")

    # Demonstration loop
    play_videos = True
    while play_videos:
        play_videos = loop
        for demo_idx in range(len(dataset_file)):
            dataset = dataset_file[f"demo_{demo_idx}"]
            fps = dataset.attrs["control_frequency"] / 2
            wrist_imgs = dataset["obs/wrist_img"]
            table_imgs = dataset["obs/table_img"]

            # Video playback loop
            for i in range(len(wrist_imgs)):
                start = time.perf_counter()
                
                cv.imshow("Demonstration", np.concat((wrist_imgs[i], table_imgs[i]), axis=0))
                cv.pollKey()

                # Limit frame rate
                elapsed = time.perf_counter() - start
                sleep_time = 1 / fps - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show videos of a converted simulation dataset creatd by create_sim_demonstrations.py."
    )
    parser.add_argument("-d", "--dataset", type=Path, required=True, help="Path to dataset created by create_sim_demonstrations.py")
    parser.add_argument("-l", "--loop", action="store_true", help="Loop through videos")

    return parser.parse_args()


def main():
    args = parse_args()
    playback_dataset_videos(args.dataset, args.loop)


if __name__ == "__main__":
    main()
