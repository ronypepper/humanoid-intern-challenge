"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
from pathlib import Path
import h5py

REPO_NAME = "ronypepper/humanoid-intern-challenge-TODO"  # Name of the output dataset, also used for the Hugging Face Hub


def main(dataset_path: Path, hf_repo_name: str, push_to_hub: bool = False):
    # Load dataset
    dataset_file = h5py.File(dataset_path, "r")

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Demonstration loop
    for demo_idx in range(len(dataset_file)):
        dataset = dataset_file[f"demo_{demo_idx}"]
        prompt = dataset.attrs["prompt"]
        joint_obss = dataset["obs/joint_pos"]
        joint_actss = dataset["act/joint_pos"]
        wrist_imgs = dataset["obs/wrist_img"]
        table_imgs = dataset["obs/table_img"]

        for i in range(joint_obss.shape[0]):
            dataset.add_frame(
                {
                    "image": table_imgs[i],
                    "wrist_image": wrist_imgs[i],
                    "state": joint_obss[i],
                    "actions": joint_actss[i],
                    "task": prompt,
                }
            )
        dataset.save_episode()            

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["humanoid-intern-challenge"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
