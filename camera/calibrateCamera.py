import cv2 as cv
import numpy as np
import glob, sys

# 1) Checkerboard settings
CHECKERBOARD = (9, 6)
square_size = 0.022  # 25 mm squares in meters

# prepare object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints, imgpoints = [], []
image_shape = None

# Termination criteria for cornerSubPix:
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

images = glob.glob('calib_images/*.jpg')
if not images:
    print("No images found in calib_images/")
    sys.exit(1)

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Can't read {fname}, skipping.")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if image_shape is None:
        image_shape = gray.shape[::-1]

    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        print(f"No board in {fname}, skipping.")
        continue

    objpoints.append(objp)
    # refine with the correct signature:
    corners2 = cv.cornerSubPix(
        gray,
        corners,
        winSize=(11,11),
        zeroZone=(-1,-1),
        criteria=criteria
    )
    imgpoints.append(corners2)

    cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv.imshow('Calibration', img)
    cv.waitKey(100)

cv.destroyAllWindows()

if not objpoints:
    print("No valid detections – check your images.")
    sys.exit(1)

print(f"Calibrating with image size {image_shape}…")
ret, K, dist, _, _ = cv.calibrateCamera(
    objpoints, imgpoints, image_shape, None, None
)

np.savez('cam_calib.npz', K=K, dist=dist)
print("Saved cam_calib.npz (K + dist).")
