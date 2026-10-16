# This script allows to perform camera calibration

import cv2 as cv
import numpy as np
import sys
import time


CALIBRATION_FILEPATH = 'demo/macbook_1920x1080_calibration.npz'
CAMERA_DEVICE_ID = 1
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080


NUM_CALIBRATION_IMAGES = 30
CAPTURE_INTERVAL = 0.5  # seconds


# Charuco calibration board configuration
pattern_size = (5, 7)
square_size = 0.030  # meters
marker_size = 0.015  # meters
aruco_dict = cv.aruco.DICT_4X4_250

# Prepare charuco board detector
aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict)
board = cv.aruco.CharucoBoard(pattern_size, square_size, marker_size, aruco_dict)
charuco_detector = cv.aruco.CharucoDetector(board)
obj_points = []  # 3D points in real world
img_points = []  # 2D points in image plane

# Open camera
# cap = cv.VideoCapture(0, cv.CAP_V4L2)
# cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
cap = cv.VideoCapture(CAMERA_DEVICE_ID)
cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
if not cap.isOpened():
    sys.exit("Error: Cannot open camera")

frame = None
num_imgs_captured = 0
img_capture_time = time.perf_counter()
while True:
    # Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        continue

    # Charuco detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    if (corners is not None and
            charuco_ids is not None and
            len(corners) == len(charuco_ids) and
            len(corners) >= 4):
        draw_corners = corners.reshape(-1, 1, 2)
        cv.aruco.drawDetectedCornersCharuco(
            frame,
            draw_corners,
            charuco_ids,
            (0, 0, 255)
        )

        if time.perf_counter() >= img_capture_time:
            frame_obj_points, frame_img_points = board.matchImagePoints(corners, charuco_ids)
            img_points.append(frame_img_points)
            obj_points.append(frame_obj_points)

            num_imgs_captured += 1
            if num_imgs_captured >= NUM_CALIBRATION_IMAGES:
                break
            img_capture_time = time.perf_counter() + CAPTURE_INTERVAL

    # Visualization
    frame = cv.flip(frame, 1)
    cv.putText(frame, f"Images remaining: {NUM_CALIBRATION_IMAGES - num_imgs_captured}", (70, 100),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow('Camera', frame)

# Calibrate camera
if frame is None:
    sys.exit("Error: Cannot calibrate camera")
print("\nCalibrating camera...")
h, w = frame.shape[:2]
reproj_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    obj_points, img_points, (w, h), None, None
)

# Print results
print(f"\nCalibration successful!")
print(f"\nRMS re-projection error: {reproj_error:.4f}")
print(f"\nCamera matrix:\n{camera_matrix}")
print(f"\nDistortion coefficients:\n{dist_coeffs.ravel()}")

# Save calibration
np.savez(CALIBRATION_FILEPATH,
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         rvecs=rvecs,
         tvecs=tvecs)
print(f"\nCalibration saved to \"{CALIBRATION_FILEPATH}\"")

# Release resources
cap.release()
cv.destroyAllWindows()
