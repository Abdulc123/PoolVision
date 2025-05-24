# hsv_calibrator.py
import cv2
import numpy as np

def hsvCalibrator(frame_function, window_prefix='HSV Calibrator'):
    """
    frame_provider: a zero-arg function that returns your warped BGR frame each time it's called.
                    Should return None (or False) to exit.
    """
    # 1) create trackbar window
    win = window_prefix
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 2) create H/S/V min/max sliders
    cv2.createTrackbar('H min', win, 0,   180, lambda x: None)
    cv2.createTrackbar('S min', win, 0,   255, lambda x: None)
    cv2.createTrackbar('V min', win, 0,   255, lambda x: None)
    cv2.createTrackbar('H max', win, 180, 180, lambda x: None)
    cv2.createTrackbar('S max', win, 255, 255, lambda x: None)
    cv2.createTrackbar('V max', win, 255, 255, lambda x: None)
    frame  = frame_function()
    while True:
        if frame is None:
            break

        # 3) convert warped to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 4) read slider positions
        h_min = cv2.getTrackbarPos('H min', win)
        s_min = cv2.getTrackbarPos('S min', win)
        v_min = cv2.getTrackbarPos('V min', win)
        h_max = cv2.getTrackbarPos('H max', win)
        s_max = cv2.getTrackbarPos('S max', win)
        v_max = cv2.getTrackbarPos('V max', win)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # 5) build mask + preview
        mask   = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Warped Input', frame)
        cv2.imshow('Mask',        mask)
        cv2.imshow('Result',      result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"LOWER_GREEN_HSV_RANGE = np.array({lower.tolist()})")
            print(f"UPPER_GREEN_HSV_RANGE = np.array({upper.tolist()})")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
