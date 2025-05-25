# debug_ball_color_tuner.py

import cv2 as cv
import numpy as np

import config.settings as Settings
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame
from processing.BallDetector import BallDetector
from display.screenDisplay import drawColoredBallPositions

# callback for empty trackbars
def nothing(x): pass

def main():
    # 1) Create a tuning window & trackbars
    cv.namedWindow('Tuner', cv.WINDOW_NORMAL)
    cv.createTrackbar('WHITE_SAT',   'Tuner',  60, 255, nothing)
    cv.createTrackbar('WHITE_VAL',   'Tuner', 200, 255, nothing)
    cv.createTrackbar('STRIPE_%',    'Tuner',  20, 100, nothing)
    cv.createTrackbar('SAMPLE_RAD',  'Tuner',  15,  50, nothing)

    # 2) Init your camera and detector as you do in main.py
    camera = CameraCapture(
        camera_index=0,
        width = Settings.WARPED_TABLE_W,
        height= Settings.WARPED_TABLE_H
    )
    detector = BallDetector(
        table_px_size   = (Settings.WARPED_TABLE_W, Settings.WARPED_TABLE_H),
        table_m_size    = (Settings.TABLE_WIDTH_M,   Settings.TABLE_HEIGHT_M),
        min_px_radius   = Settings.MIN_BALL_RADIUS_PX,
        max_px_radius   = Settings.MAX_BALL_RADIUS_PX,
        green_hsv_range = (Settings.LOWER_GREEN_HSV_RANGE, Settings.UPPER_GREEN_HSV_RANGE)
    )

    while True:
        # grab & warp
        frame  = camera.get_frame()
        warped = getWarpedFrame(frame, debug_mode=False)
        if warped is None:
            continue

        # 3) read trackbar values each loop
        ws = cv.getTrackbarPos('WHITE_SAT',  'Tuner')
        wv = cv.getTrackbarPos('WHITE_VAL',  'Tuner')
        sp = cv.getTrackbarPos('STRIPE_%',   'Tuner') / 100.0
        sr = cv.getTrackbarPos('SAMPLE_RAD','Tuner')

        # override detector thresholds
        detector.WHITE_SAT_THRESH   = ws
        detector.WHITE_VAL_THRESH   = wv
        detector.STRIPE_FRAC_THRESH = sp

        # 4) run detection
        raw_centers    = detector.rawDetectBalls(warped)
        colored_result = detector.findColorsOfBalls(warped, raw_centers, radius=sr)

        # 5) draw & show
        vis = warped.copy()
        # draw raw centers in gray
        for (x,y) in raw_centers:
            cv.circle(vis, (x,y), sr, (200,200,200), 1)
        drawColoredBallPositions(colored_result, vis)

        cv.imshow('Tuner', vis)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
