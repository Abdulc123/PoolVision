import cv2
import numpy as np
from cv2 import aruco

# 1) Pick your dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

marker_size_px = 400  # size of each marker in pixels
for marker_id in range(6):
    # generate the marker image
    marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

    # save it
    fname = f"aruco_marker_{marker_id}.png"
    cv2.imwrite(fname, marker_img)
    print(f"Saved {fname}")
