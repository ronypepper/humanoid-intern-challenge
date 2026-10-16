from dataclasses import dataclass, field
import mediapipe as mp
import numpy as np
import cv2 as cv

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark as HandKeypointIndices


@dataclass(frozen=True)
class HandKeypointsTrackerConfig:
    model_asset_path: str = "hand_landmarker.task"
    camera_a_calibration_path: str = "calibration_a.npz"
    camera_b_calibration_path: str = "calibration_b.npz"
    cam_a_T: np.ndarray = field(default_factory=lambda: np.eye(4))
    cam_b_T: np.ndarray = field(default_factory=lambda: np.eye(4))


class HandKeypointsTracker:
    def __init__(self, cfg: HandKeypointsTrackerConfig):
        # Load camera calibrations
        with np.load(cfg.camera_a_calibration_path) as calibration:
            cam_a_K = calibration['camera_matrix']
            cam_a_dist_coeffs = calibration['dist_coeffs']
        self.cam_a_proj = cam_a_K @ cfg.cam_a_T[0:3]
        with np.load(cfg.camera_b_calibration_path) as calibration:
            cam_b_K = calibration['camera_matrix']
            cam_b_dist_coeffs = calibration['dist_coeffs']
        self.cam_b_proj = cam_b_K @ cfg.cam_b_T[0:3]
        
        # Create hand landmarker detectors
        base_options = mp.tasks.BaseOptions(model_asset_path=cfg.model_asset_path)
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                                        running_mode=mp.tasks.vision.RunningMode.VIDEO,
                                                        num_hands=2)
        self.detector_a = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.detector_b = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _triangulate_3d_keypoints(self, 
                                  landmarks_a: list[NormalizedLandmark], 
                                  landmarks_b: list[NormalizedLandmark], 
                                  width: int, 
                                  height: int) -> np.ndarray:
        # Transform normalized image-space 2D landmarks to pixel-space
        landmarks_a = [(l.x, l.y) for l in landmarks_a]
        landmarks_a = np.array(landmarks_a) * (width - 1, height - 1)
        landmarks_a = np.clip(landmarks_a, (0, 0), (width - 1, height - 1))

        landmarks_b = [(l.x, l.y) for l in landmarks_b]
        landmarks_b = np.array(landmarks_b) * (width - 1, height - 1)
        landmarks_b = np.clip(landmarks_b, (0, 0), (width - 1, height - 1))

        # Triangulate 3D keypoint positions
        landmarks_4D = cv.triangulatePoints(self.cam_a_proj, self.cam_b_proj, landmarks_a.T, landmarks_b.T)
        landmarks_3D = landmarks_4D[:3] / landmarks_4D[3:4]
        landmarks_3D = landmarks_3D.T  # (N, 3)

        return landmarks_3D

    def detect_hand_keypoints(self, frame_a: cv.typing.MatLike, 
                              frame_b: cv.typing.MatLike, 
                              timestamp_a_ms: int, 
                              timestamp_b_ms: int, 
                              annotate: bool = False) -> tuple[dict[str, np.ndarray], cv.typing.MatLike, cv.typing.MatLike]:
        # Preprocess frames
        height, width, _ = frame_a.shape
        if frame_b.shape[0] != height or frame_b.shape[1] != width:
            raise ValueError("HandLandmarksTracker: frame_a and frame_b have different dimensions!")
        frame_a = cv.cvtColor(frame_a, cv.COLOR_BGR2RGB)
        frame_b = cv.cvtColor(frame_b, cv.COLOR_BGR2RGB)
        mp_image_a = mp.Image(mp.ImageFormat.SRGB, frame_a)
        mp_image_b = mp.Image(mp.ImageFormat.SRGB, frame_b)

        # Detect 2D hand landmarks
        result_a = self.detector_a.detect_for_video(mp_image_a, timestamp_a_ms)
        result_b = self.detector_b.detect_for_video(mp_image_b, timestamp_b_ms)

        # Annotate frame
        annotated_frame_a, annotated_frame_b = frame_a, frame_b
        if annotate:
            annotated_frame_a = np.copy(frame_a)
            annotated_frame_b = np.copy(frame_b)
            for frame, result in zip((annotated_frame_a, annotated_frame_b), (result_a, result_b)):
                for landmarks in result.hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())        

        # Compute 3D keypoints of hands if possible
        if len(result_a.hand_landmarks) == 0 or len(result_b.hand_landmarks) == 0:
            return ({}, annotated_frame_a, annotated_frame_b)
        handedness_a = [h[0].category_name for h in result_a.handedness]
        handedness_b = [h[0].category_name for h in result_b.handedness]
        hand_keypoints = {}
        if "Left" in handedness_a and "Left" in handedness_b:
            landmarks_a = result_a.hand_landmarks[handedness_a.index("Left")]
            landmarks_b = result_b.hand_landmarks[handedness_b.index("Left")]
            hand_keypoints["Left"] = self._triangulate_3d_keypoints(landmarks_a, landmarks_b, width, height)
        if "Right" in handedness_a and "Right" in handedness_b:
            landmarks_a = result_a.hand_landmarks[handedness_a.index("Right")]
            landmarks_b = result_b.hand_landmarks[handedness_b.index("Right")]
            hand_keypoints["Right"] = self._triangulate_3d_keypoints(landmarks_a, landmarks_b, width, height)

        return hand_keypoints, annotated_frame_a, annotated_frame_b

    def __del__(self):
        self.detector_a.close()
        self.detector_b.close()
