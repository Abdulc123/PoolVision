# camera/warp.py

import cv2
import numpy as np
from cv2 import aruco

# 1) CONFIGURE YOUR TABLE “CANVAS” SIZE IN PIXELS
TABLE_W, TABLE_H = 1000, 500

# 2) MAP each marker ID → its desired (x,y) in that top-down view
#    IDs 0–3 are corners, 4=top-middle, 5=bottom-middle
DEST_PT = {
    0: [0,         0],
    1: [TABLE_W,   0],
    2: [TABLE_W,   TABLE_H],
    3: [0,         TABLE_H],
    4: [TABLE_W//2, 0],
    5: [TABLE_W//2, TABLE_H],
}

# 3) SET UP your ArUco detector once
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()

def get_warped_frame(frame):
    """
    - frame: BGR image from your capture device
    → returns: top-down warped BGR frame of size (TABLE_W, TABLE_H)
               or None if fewer than 4 of your markers are visible.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is None:
        return None

    ids = ids.flatten()
    src_pts, dst_pts = [], []

    # ONLY use corner markers (0,1,2,3) for homography
    for c, mid in zip(corners, ids):
        if mid in (0, 1, 2, 3):
            centre = c[0].mean(axis=0)
            src_pts.append(centre)
            dst_pts.append(DEST_PT[mid])

    # need exactly 4 corners
    if len(src_pts) != 4:
        return None

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or H.shape != (3, 3):
        # failed to compute a valid homography
        print("Homography estimation failed")
        return None

    # ensure correct dtype
    H = H.astype(np.float32)

    warped = cv2.warpPerspective(frame, H, (TABLE_W, TABLE_H))
    return warped