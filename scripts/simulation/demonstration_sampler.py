import h5py
from scipy.spatial.transform import Rotation

from robosuite.utils.placement_samplers import ObjectPositionSampler


class DemonstrationSampler(ObjectPositionSampler):
    def __init__(self, name, dataset_file: h5py.File, table_offset, z_offset=0.0):
        super().__init__(name=name, z_offset=z_offset)
        self.dataset_file = dataset_file
        self.table_offset = table_offset
        self.demo_idx = 0

    def reset(self):
        super().reset()
        self.demo_idx = 0

    def sample(self, fixtures=None, reference=None, on_top=True):
        if len(self.dataset_file) <= self.demo_idx:
            raise RuntimeError("DemonstrationSampler: reached end of dataset!")

        objects_dataset = self.dataset_file[f"demo_{self.demo_idx}/objects"]
        self.demo_idx += 1

        placed_objects = {}
        for obj in self.mujoco_objects:
            if obj.name in objects_dataset:
                T = objects_dataset[obj.name]["initial_pose"]
                pos = T[:3, 3] + self.table_offset
                quat = Rotation.from_matrix(T[:3, :3]).as_quat()
                placed_objects[obj.name] = (pos, quat, obj)

        return placed_objects
