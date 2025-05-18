import numpy as np
import cv2 as cv
# These dimensions define the size of the output warped image (top-down view of the table).
WARPED_TABLE_W, WARPED_TABLE_H = 1280, 720  # Width and height of the table in pixels


# MAP each marker ID → its desired (x, y) in that top-down view
# This dictionary maps ArUco marker IDs to their corresponding positions in the warped (top-down) view.
# Marker IDs 0–3 are used as the corners of the table.
WARPED_DEST_PT = {
    0: [0,         0],          # Top-left corner of the table
    1: [WARPED_TABLE_W,   0],          # Top-right corner of the table
    2: [WARPED_TABLE_W,   WARPED_TABLE_H],    # Bottom-right corner of the table
    3: [0,         WARPED_TABLE_H],    # Bottom-left corner of the table
}

# CAMERA CALIBRATION VARIABLES
# 0) load your saved calibration
data = np.load('cam_calib.npz')
K, dist = data['K'], data['dist']

# 1) Load your one-time calibration results:
data = np.load('cam_calib.npz')
K, dist = data['K'], data['dist']

# 2) Build optimal new camera matrix + undistort maps at your capture size
new_K, _ = cv.getOptimalNewCameraMatrix(K, dist,
                                        (WARPED_TABLE_W, WARPED_TABLE_H),
                                        alpha=1,
                                        newImgSize=(WARPED_TABLE_W, WARPED_TABLE_H))
map1, map2 = cv.initUndistortRectifyMap(K, dist, None,
                                        new_K,
                                        (WARPED_TABLE_W, WARPED_TABLE_H),
                                        cv.CV_16SC2)