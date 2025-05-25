import cv2 as cv
import numpy as np
from config.settings import LOWER_GREEN_HSV_RANGE, UPPER_GREEN_HSV_RANGE
from camera.TableWarper import getWarpedFrame
from camera.capture import CameraCapture

def nothing(x): pass

# ——————————————————————————————————————————————————
# 1) Create your parameter window and trackbars
# ——————————————————————————————————————————————————
cv.namedWindow("Params", cv.WINDOW_NORMAL)
cv.createTrackbar("Kernel",    "Params", 5,   30, nothing)  # must be odd
cv.createTrackbar("MinRad",    "Params", 10,  100, nothing)
cv.createTrackbar("MaxRad",    "Params", 50,  200, nothing)
cv.createTrackbar("dp*10",     "Params", 10,  30,  nothing)  # Hough inv ratio ×10
cv.createTrackbar("MinDist",   "Params", 50,  500, nothing)
cv.createTrackbar("CannyHi",   "Params", 100, 300, nothing)
cv.createTrackbar("AccThresh", "Params", 30,  100, nothing)

# prepare capture
cap = CameraCapture(camera_index=0,
                    width=1280, height=720)

cv.namedWindow("Mask", cv.WINDOW_NORMAL)
cv.namedWindow("Detections", cv.WINDOW_NORMAL)

while True:
    frame = cap.get_frame()
    warped   = getWarpedFrame(frame, debug_mode=False)
    hsv       = cv.cvtColor(warped, cv.COLOR_BGR2HSV)

    # 2) mask out the table and invert → balls are white
    table_mask = cv.inRange(hsv,
                            LOWER_GREEN_HSV_RANGE,
                            UPPER_GREEN_HSV_RANGE)
    ball_mask = cv.bitwise_not(table_mask)

    # 3) get params from trackbars
    ksz   = cv.getTrackbarPos("Kernel", "Params") // 2 * 2 + 1
    minR  = cv.getTrackbarPos("MinRad", "Params")
    maxR  = cv.getTrackbarPos("MaxRad", "Params")
    dp    = cv.getTrackbarPos("dp*10", "Params") / 10.0
    minD  = cv.getTrackbarPos("MinDist", "Params")
    hi    = cv.getTrackbarPos("CannyHi", "Params")
    accTh = cv.getTrackbarPos("AccThresh", "Params")

    # 4) clean up the mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksz, ksz))
    proc = cv.morphologyEx(ball_mask, cv.MORPH_OPEN,  kernel, iterations=1)
    proc = cv.morphologyEx(proc,      cv.MORPH_CLOSE, kernel, iterations=1)

    # 5) run HoughCircles
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (ksz, ksz), 0)
    circles = cv.HoughCircles(blur,
                              cv.HOUGH_GRADIENT,
                              dp=dp,
                              minDist=minD,
                              param1=hi,
                              param2=accTh,
                              minRadius=minR,
                              maxRadius=maxR)

    # 6) visualize
    disp = warped.copy()
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            cv.circle(disp, (x, y), r, (0, 255, 255), 2)
            cv.circle(disp, (x, y), 2, (0, 0, 255), 3)

    cv.imshow("Mask", proc)
    cv.imshow("Detections", disp)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
