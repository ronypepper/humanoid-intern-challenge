from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2 as cv
import h5py
import time

from charuco_board_tracker import CharucoBoardConfig, CharucoBoardsTracker
from hand_keypoints_tracker import HandKeypointIndices, HandKeypointsTrackerConfig, HandKeypointsTracker
from utils import rotation_matrix_from_directions, transform_matrix_from_vectors


@dataclass(frozen=True)
class CameraConfig:
    camera_device_id: int = 0
    camera_width: int = 1920
    camera_height: int = 1080
    camera_calibration_path: str = "calibration.npz"


@dataclass(frozen=True)
class DemoRecorderConfig:
    camera_hands_a: CameraConfig = field(default_factory=CameraConfig)
    camera_hands_b: CameraConfig = field(default_factory=CameraConfig)
    camera_objects: CameraConfig = field(default_factory=CameraConfig)
    origin: CharucoBoardConfig = field(default_factory=CharucoBoardConfig)
    origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_hand: Literal["Left", "Right"] = "Left"
    handlandmarker_model_asset_path: str = "hand_landmarker.task"
    visualize_hands: bool = False


class DemoRecorder:
    def __init__(self, cfg: DemoRecorderConfig):
        self.cfg = cfg
        self.hand_pose_trigger_timestamp = None

        # Initialize cameras
        self.cap_hands_a = self._initialize_camera(cfg.camera_hands_a)
        self.cap_hands_b = self._initialize_camera(cfg.camera_hands_b)
        if self.cfg.camera_objects.camera_device_id == self.cfg.camera_hands_a.camera_device_id:
            self.cap_objects = self.cap_hands_a
        elif self.cfg.camera_objects.camera_device_id == self.cfg.camera_hands_b.camera_device_id:
            self.cap_objects = self.cap_hands_b
        else:
            self.cap_objects = self._initialize_camera(cfg.camera_objects)
        
        # Initialize object tracking
        self.object_detector = CharucoBoardsTracker(self.cfg.camera_objects.camera_calibration_path)
        self.object_origin_offsets = {}

    def _initialize_camera(self, cfg: CameraConfig) -> cv.VideoCapture:
        # Open and configure camera
        cap = cv.VideoCapture(cfg.camera_device_id)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
        if not cap.isOpened():
            raise RuntimeError(f"DemoRecorder: cannot open camera {cfg.camera_device_id}!")
        if cap.get(cv.CAP_PROP_FRAME_WIDTH) != cfg.camera_width:
            raise RuntimeError(f"DemoRecorder: failed to set width {cfg.camera_width} for camera {cfg.camera_device_id}!")
        if cap.get(cv.CAP_PROP_FRAME_HEIGHT) != cfg.camera_height:
            raise RuntimeError(f"DemoRecorder: failed to set width {cfg.camera_height} for camera {cfg.camera_device_id}!")
        return cap

    def _on_enter_pressed(self, vis):
        self.user_pressed_enter = True
        return False

    def _initialize_3d_visualizer(self, object_poses : dict[str, np.ndarray]) -> None:        
        self.vis = o3d.visualization.Visualizer()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.user_pressed_enter = False
        self.vis.register_key_callback(
            257,  # GLFW_KEY_ENTER
            lambda vis: self._on_enter_pressed(vis)
        )
        self.vis.create_window(window_name="Demonstration", width=1920, height=1080, left=1920, top=50)

        # Add world origin coordinate system axis
        world_origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        self.vis.add_geometry(world_origin_axis)

        # Add coordinate system axis for gripper pose
        self.vis_gripper_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self.vis_gripper_axis_pose = np.eye(4)
        self.vis.add_geometry(self.vis_gripper_axis)

        if self.cfg.visualize_hands:
            # Add spheres for hand keypoints + gripper position
            self.vis_hand_keypoints = [o3d.geometry.TriangleMesh.create_sphere(radius=0.004) for _ in range(len(HandKeypointIndices))]
            self.vis_hand_keypoints[HandKeypointIndices.INDEX_FINGER_TIP].paint_uniform_color([1.0, 0.0, 0.0])
            self.vis_hand_keypoints[HandKeypointIndices.THUMB_TIP].paint_uniform_color([1.0, 0.0, 0.0])
            self.vis_hand_keypoints.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.008))
            self.vis_hand_keypoints[-1].paint_uniform_color([0.0, 1.0, 0.0])
            self.vis_hand_keypoints_pos = np.zeros((len(self.vis_hand_keypoints), 3))
            for sphere in self.vis_hand_keypoints:
                sphere.compute_vertex_normals()
                self.vis.add_geometry(sphere)

        # Add coordinate system axis at initial object poses
        for pose in object_poses.values():
            coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            coord_axis.transform(pose)
            self.vis.add_geometry(coord_axis)

        # Set view
        view_ctrl = self.vis.get_view_control()
        view_ctrl.set_lookat([0, 0, 0])
        view_ctrl.set_front([1, 1, 1.5])
        view_ctrl.set_up([0, 0, 1])
        view_ctrl.set_zoom(1.2)

    def add_tracked_object(self, name: str, cfg: CharucoBoardConfig, origin_offset: tuple[float, float, float]) -> None:
        self.object_detector.add_board(name, cfg)
        self.object_origin_offsets[name] = origin_offset

    def _apply_origin_offset(self, origin_T: np.ndarray, origin_offset: tuple[float, float, float]):
        tvec = origin_T[:3, 3] + origin_T[:3, :3] @ np.array(origin_offset)
        Rx, _ = cv.Rodrigues(np.array([np.pi, 0, 0]))
        R = origin_T[:3, :3] @ Rx
        rvec, _ = cv.Rodrigues(R)
        rvec = rvec.flatten()
        T = transform_matrix_from_vectors(rvec, tvec)
        return T, rvec, tvec

    def _detect_object_poses(self) -> dict[str, np.ndarray]:
        charuco_detector = CharucoBoardsTracker(self.cfg.camera_objects.camera_calibration_path)
        charuco_detector.add_board("origin", self.cfg.origin)

        # Loop until origin and object charuco boards have been detected and accepted by the user
        while True:
            # Read frame
            ret, frame = self.cap_objects.read()
            if not ret:
                print("DemoRecorder: cannot read frame from camera...")
                continue

            # Detect origin (board)
            boards_T = charuco_detector.detect_board_poses(frame)
            if len(boards_T) == 0:
                frame_with_text = cv.putText(np.copy(frame), "Origin not detected...", 
                                             (70, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imshow("Demo Recorder", frame_with_text)
                cv.waitKey(1)
                continue
            origin_T = boards_T["origin"]
            origin_T, rvec, tvec = self._apply_origin_offset(origin_T, self.cfg.origin_offset)
            annotated_frame = np.copy(frame)
            cv.drawFrameAxes(annotated_frame, charuco_detector.cam_K, charuco_detector.dist_coeffs, rvec, tvec, 0.3)

            # Detect tracked objects (boards)
            boards_T = self.object_detector.detect_board_poses(frame)
            if len(boards_T) == 0:
                frame_with_text = cv.putText(annotated_frame, "No boards detected...", 
                                             (70, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imshow("Demo Recorder", frame_with_text)
                cv.waitKey(1)
                continue

            for name in boards_T.keys():
                boards_T[name], rvec, tvec = self._apply_origin_offset(boards_T[name], self.object_origin_offsets[name])
                cv.drawFrameAxes(annotated_frame, charuco_detector.cam_K, charuco_detector.dist_coeffs, rvec, tvec, 0.3)

            # Query user to accept tracked objects
            frame_with_text = cv.putText(np.copy(annotated_frame), "Press Enter to accept tracked object poses...", 
                                         (70, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imshow("Demo Recorder", frame_with_text)
            key = cv.waitKey(1) & 0xFF
            if key == 13:  # Enter
                frame_with_text = cv.putText(annotated_frame, "Press Enter to confirm or Escape to cancel...", 
                                             (70, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imshow("Demo Recorder", frame_with_text)
                while True:
                    key = cv.waitKey(0) & 0xFF
                    if key == 13:  # Enter
                        # Transform object poses to be relative to origin
                        origin_T_inv = np.linalg.inv(origin_T)
                        objects_T = {name : origin_T_inv @ T for name, T in boards_T.items()}
                        return objects_T
                    elif key == 27:  # Escape
                        break

    def _detect_camera_pose(self, cfg: CameraConfig, cap: cv.VideoCapture) -> np.ndarray:
        # Loop until origin charuco board has been detected and accepted by the user
        print("DemoRecorder: detecting origin...")
        charuco_detector = CharucoBoardsTracker(cfg.camera_calibration_path)
        charuco_detector.add_board("origin", self.cfg.origin)
        while True:            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("DemoRecorder: cannot read frame from camera...")
                continue

            # Detect board
            boards_T = charuco_detector.detect_board_poses(frame)
            if len(boards_T) == 0:
                frame_with_text = cv.putText(np.copy(frame), "Origin not detected...", 
                                             (70, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imshow("Demo Recorder", frame_with_text)
                cv.waitKey(1)
                continue
            origin_T = boards_T["origin"]
            origin_T, rvec, tvec = self._apply_origin_offset(origin_T, self.cfg.origin_offset)
            annotated_frame = np.copy(frame)
            cv.drawFrameAxes(annotated_frame, charuco_detector.cam_K, charuco_detector.dist_coeffs, rvec, tvec, 0.3)

            # Query user to accept origin
            frame_with_text = cv.putText(np.copy(annotated_frame), "Press Enter to set world origin...", 
                                         (70, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imshow("Demo Recorder", frame_with_text)
            key = cv.waitKey(1) & 0xFF
            if key == 13:  # Enter
                frame_with_text = cv.putText(annotated_frame, "Press Enter to confirm or Escape to cancel...", 
                                             (70, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imshow("Demo Recorder", frame_with_text)
                while True:
                    key = cv.waitKey(0) & 0xFF
                    if key == 13:  # Enter
                        return origin_T
                    elif key == 27:  # Escape
                        break

    def _compute_gripper_state(self, hand_keypoints: dict[str, np.ndarray]) -> tuple[bool, np.ndarray, float]:
        if self.cfg.gripper_hand not in hand_keypoints:
            return False, np.eye(4), 0.0
        keypoints = hand_keypoints[self.cfg.gripper_hand]

        # Compute gripper position
        index_tip_pos = keypoints[HandKeypointIndices.INDEX_FINGER_TIP]
        thumb_tip_pos = keypoints[HandKeypointIndices.THUMB_TIP]
        gripper_pos = (index_tip_pos + thumb_tip_pos) / 2

        # Compute gripper orientation
        _DIRECTION_INDICES = [HandKeypointIndices.THUMB_CMC, HandKeypointIndices.THUMB_MCP, HandKeypointIndices.THUMB_IP,
                              HandKeypointIndices.INDEX_FINGER_MCP, HandKeypointIndices.INDEX_FINGER_PIP, HandKeypointIndices.INDEX_FINGER_DIP]
        gripper_z_dir = np.zeros_like(gripper_pos)
        for idx in _DIRECTION_INDICES:
            gripper_z_dir += gripper_pos - keypoints[idx]
        gripper_y_dir = thumb_tip_pos - index_tip_pos
        try:
            gripper_R = rotation_matrix_from_directions(gripper_z_dir, gripper_y_dir)
            Rz, _ = cv.Rodrigues(np.array([0, 0, np.pi / 2]))
            gripper_R = gripper_R @ Rz
        except ValueError:
            return False, np.eye(4), 0.0

        # Combine to transform matrix
        gripper_T = np.eye(4)
        gripper_T[:3, :3] = gripper_R
        gripper_T[:3, 3] = gripper_pos

        # Compute gripper state
        gripper_opening = np.linalg.norm(gripper_y_dir)

        return True, gripper_T, gripper_opening

    def _detect_hand_pose_trigger(self, hand_keypoints: dict[str, np.ndarray]) -> bool:
        if self.cfg.gripper_hand not in hand_keypoints:
            self.hand_pose_trigger_timestamp = None
            return False
        
        _TIP_INDICES = [HandKeypointIndices.THUMB_TIP, 
                        HandKeypointIndices.INDEX_FINGER_TIP, 
                        HandKeypointIndices.MIDDLE_FINGER_TIP, 
                        HandKeypointIndices.RING_FINGER_TIP, 
                        HandKeypointIndices.PINKY_TIP]
        keypoints = hand_keypoints[self.cfg.gripper_hand][_TIP_INDICES]
        center = np.sum(keypoints) / len(_TIP_INDICES)
        tip_center_avg_distance = np.average(np.linalg.norm(center - keypoints, axis=1))
        if tip_center_avg_distance > 0.02:
            self.hand_pose_trigger_timestamp = None
        else:
            if self.hand_pose_trigger_timestamp is None:
                self.hand_pose_trigger_timestamp = time.time()
            elif self.hand_pose_trigger_timestamp + 2.0 <= time.time():
                self.hand_pose_trigger_timestamp = None
                return True
        return False

    def record_demonstration(self, prompt: str, dataset_path: Path):
        # Initialize dataset
        dataset_file = h5py.File(dataset_path, "a")
        demo_idx = 0
        while f"demo_{demo_idx}" in dataset_file.keys():
            demo_idx += 1
        dataset = dataset_file.create_group(f"demo_{demo_idx}")
        dataset.attrs["prompt"] = prompt

        # Detect objects and save their initial poses to the dataset
        objects_names = []
        if self.object_detector.get_num_boards() > 0:  # check if no objects have been registered
            print("DemoRecorder: detecting initial object poses...")
            objects_T = self._detect_object_poses()
            for name, T in objects_T.items():
                dataset[f"objects/{name}/initial_pose"] = T
                objects_names.append(name)
            print(f"DemoRecorder: {len(objects_T)} objects detected...")
        else:
            print("DemoRecorder: no objects registered, skipping detection.")
        
        # Detect hand tracking camera poses
        # This is done here because I reuse one hand tracking camera for object detection above, for 
        # which I physically move the camera. A redetection of its pose is therefore required for every demo.
        cam_a_T = self._detect_camera_pose(self.cfg.camera_hands_a, self.cap_hands_a)
        cam_b_T = self._detect_camera_pose(self.cfg.camera_hands_b, self.cap_hands_b)

        # Initialize hand tracking
        hand_tracker_cfg = HandKeypointsTrackerConfig(self.cfg.handlandmarker_model_asset_path,
                                                      self.cfg.camera_hands_a.camera_calibration_path,
                                                      self.cfg.camera_hands_b.camera_calibration_path,
                                                      cam_a_T,
                                                      cam_b_T)
        self.hand_tracker = HandKeypointsTracker(hand_tracker_cfg)

        # Initialize visualization
        cv.destroyWindow("Demo Recorder")
        self._initialize_3d_visualizer(objects_T)

        # Demonstration recording loop
        init_timestamp_cam_a_ns, init_timestamp_cam_b_ns = None, None
        self.user_pressed_enter = False
        demo_recording_started = False
        demo_gripper_T, demo_gripper_opening, demo_gripper_timing = [], [], []
        while True:
            # Load next camera frames (separate grab/retrieve for better frame simultaneity)
            ok_a = self.cap_hands_a.grab()
            timestamp_cam_a_ns = time.perf_counter_ns()
            ok_b = self.cap_hands_b.grab()
            timestamp_cam_b_ns = time.perf_counter_ns()
            if not (ok_a and ok_b):
                continue
            ok_a, frame_a = self.cap_hands_a.retrieve()
            ok_b, frame_b = self.cap_hands_b.retrieve()
            if not (ok_a and ok_b):
                continue
            if init_timestamp_cam_a_ns is None:
                init_timestamp_cam_a_ns = timestamp_cam_a_ns
                init_timestamp_cam_b_ns = timestamp_cam_b_ns
            timestamp_cam_a_ns -= init_timestamp_cam_a_ns
            timestamp_cam_b_ns -= init_timestamp_cam_b_ns
            frame_timestamp = (timestamp_cam_a_ns + timestamp_cam_b_ns) / 2 / 1000_0000_000

            # Infer gripper state from tracked hand
            ret = self.hand_tracker.detect_hand_keypoints(frame_a, 
                                                          frame_b, 
                                                          int(timestamp_cam_a_ns / 1_000_000), 
                                                          int(timestamp_cam_b_ns / 1_000_000),
                                                          self.cfg.visualize_hands)
            hand_keypoints, annotated_frame_a, annotated_frame_b = ret
            gripper_ok, gripper_T, gripper_opening = self._compute_gripper_state(hand_keypoints)

            # Update 3D visualizer if enabled
            if self.cfg.visualize_hands:
                # Show hand landmarks detection
                annotated_frame_a = cv.cvtColor(annotated_frame_a, cv.COLOR_RGB2BGR)
                annotated_frame_b = cv.cvtColor(annotated_frame_b, cv.COLOR_RGB2BGR)
                if demo_recording_started:
                    cv.putText(annotated_frame_a, "Recording...", 
                               (70, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imshow("Demo Recorder", np.concat((annotated_frame_a, annotated_frame_b), axis=0))

                # Update hand visualization
                if gripper_ok:
                    new_keypoint_pos = np.concat((hand_keypoints[self.cfg.gripper_hand], gripper_T[None, :3, 3]), axis=0)
                    translations = new_keypoint_pos - self.vis_hand_keypoints_pos
                    self.vis_hand_keypoints_pos = np.copy(new_keypoint_pos)
                    for i, keypoint in enumerate(self.vis_hand_keypoints):
                        keypoint.translate(translations[i])
                        self.vis.update_geometry(keypoint)

            # Update gripper pose coordiante system axis
            self.vis_gripper_axis.transform(gripper_T @ np.linalg.inv(self.vis_gripper_axis_pose))
            self.vis_gripper_axis_pose = np.copy(gripper_T)
            self.vis.update_geometry(self.vis_gripper_axis)

            self.vis.update_renderer()
            self.vis.poll_events()

            # Demo start / store data / stop logic
            if not demo_recording_started:
                # Check if user signalled demo start
                if self.user_pressed_enter:  # Enter
                    self.user_pressed_enter = False
                    demo_recording_started = True
            else:
                # Store gripper state for dataset
                if gripper_ok:
                    demo_gripper_T.append(gripper_T)
                    demo_gripper_opening.append(gripper_opening)
                    demo_gripper_timing.append(frame_timestamp)

                # Check if user signalled demo stop
                if self.user_pressed_enter:  # Enter
                    break

        # Save gripper state to dataset
        dataset["gripper/transform"] = demo_gripper_T
        dataset["gripper/opening"] = demo_gripper_opening
        dataset["gripper/timing"] = demo_gripper_timing

        # Cleanup
        dataset_file.close()
        cv.destroyWindow("Demo Recorder")
        self.vis.destroy_window()

    def __del__(self):
        self.cap_hands_a.release()
        self.cap_hands_b.release()
        self.cap_objects.release()
