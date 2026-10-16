from dataclasses import dataclass
from math import floor
import cv2 as cv
import numpy as np

from utils import transform_matrix_from_vectors


ARUCO_DICTIONARIES = {
    "DICT_4X4_50" : cv.aruco.DICT_4X4_50,
    "DICT_4X4_100" : cv.aruco.DICT_4X4_100,
    "DICT_4X4_250" : cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000" : cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50" : cv.aruco.DICT_5X5_50,
    "DICT_5X5_100" : cv.aruco.DICT_5X5_100,
    "DICT_5X5_250" : cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000" : cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50" : cv.aruco.DICT_6X6_50,
    "DICT_6X6_100" : cv.aruco.DICT_6X6_100,
    "DICT_6X6_250" : cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000" : cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50" : cv.aruco.DICT_7X7_50,
    "DICT_7X7_100" : cv.aruco.DICT_7X7_100,
    "DICT_7X7_250" : cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000" : cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL" : cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5" : cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9" : cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10" : cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11" : cv.aruco.DICT_APRILTAG_36h11,
    "DICT_ARUCO_MIP_36h12" : cv.aruco.DICT_ARUCO_MIP_36h12
}


@dataclass(frozen=True)
class CharucoBoardConfig:
    pattern_size: tuple[int, int] = (5, 5)  # (columns, rows)
    square_size: float = 0.02  # meters
    marker_size: float = 0.01  # meters
    aruco_dict: str = "DICT_4X4_250"
    aruco_dict_offset: int = 0  # ID of first aruco marker in board


class CharucoBoardsTracker:
    def __init__(self, camera_calibration_path: str = "calibration.npz"):
        self.charuco_boards = {}
        self.aruco_dict_id = None
        self.aruco_dictionary, self.aruco_detector = None, None
        
        # Load camera calibration
        with np.load(camera_calibration_path) as calibration:
            self.cam_K = calibration["camera_matrix"]
            self.dist_coeffs = calibration["dist_coeffs"]

    def add_board(self, name: str, cfg: CharucoBoardConfig):
        if name in self.charuco_boards:
            raise ValueError(f"CharucoBoardsTracker: tried to add board \"{name}\" twice!")

        # Prepare aruco detector
        if cfg.aruco_dict not in ARUCO_DICTIONARIES:
            raise ValueError(f"CharucoBoardsTracker: Unkown aruco dictionary \"{cfg.aruco_dict}\"!"
                             f"Possible values are: {list(ARUCO_DICTIONARIES.keys())}")
        if self.aruco_dict_id is None:
            self.aruco_dict_id = ARUCO_DICTIONARIES[cfg.aruco_dict]
            self.aruco_dictionary = cv.aruco.getPredefinedDictionary(self.aruco_dict_id)
            self.aruco_detector = cv.aruco.ArucoDetector(self.aruco_dictionary)
        else:
            if self.aruco_dict_id != ARUCO_DICTIONARIES[cfg.aruco_dict]:
                raise ValueError("CharucoBoardsTracker: tried to add boards of different dictionaries!")
        
        # Prepare charuco detector
        num_aruco_markers = floor(cfg.pattern_size[0] * cfg.pattern_size[1] / 2)
        aruco_ids = np.array([cfg.aruco_dict_offset + i for i in range(num_aruco_markers)], dtype=np.int32)
        board = cv.aruco.CharucoBoard(cfg.pattern_size, cfg.square_size, cfg.marker_size, self.aruco_dictionary, aruco_ids)
        detector = cv.aruco.CharucoDetector(board)

        self.charuco_boards[name] = (board, detector)

    def remove_board(self, name: str):
        if name not in self.charuco_boards:
            raise ValueError(f"CharucoBoardsTracker: tried to remove unknown board \"{name}\"!")
        self.charuco_boards.pop(name)

    def get_num_boards(self) -> int:
        return len(self.charuco_boards)

    def detect_board_poses(self, 
                           frame: cv.typing.MatLike, 
                           annotate: cv.typing.MatLike | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        # Detect aruco markers
        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(frame)
        if marker_ids is None or len(marker_ids) < 1:
            return {}
        if annotate is not None:
            cv.aruco.drawDetectedMarkers(annotate, marker_corners, marker_ids)

        # Detect charuco boards and estimate poses
        boards_T = {}
        for name, (board, detector) in self.charuco_boards.items():
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(frame, markerCorners=marker_corners, markerIds=marker_ids)
            if charuco_ids is not None and len(charuco_ids) > 0:
                if annotate is not None:
                    cv.aruco.drawDetectedCornersCharuco(annotate, charuco_corners.reshape(-1, 1, 2), charuco_ids, (0, 0, 255))
                if len(charuco_ids) >= 4:
                    try:
                        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                        flag, rvec, tvec = cv.solvePnP(obj_points, img_points, self.cam_K, self.dist_coeffs, flags=cv.SOLVEPNP_IPPE)
                        if flag:
                            boards_T[name] = transform_matrix_from_vectors(rvec, tvec)
                            if annotate is not None:
                                cv.drawFrameAxes(annotate, self.cam_K, self.dist_coeffs, rvec, tvec, 0.3)
                    except cv.error as e:
                        print("CharucoBoardsTracker: error while detecting charuco board:")
                        print(e.err)

        return boards_T
