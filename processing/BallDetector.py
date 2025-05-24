import cv2 as cv
import numpy as np

from config.settings import (
    HOUGH_KERNEL_SIZE, HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS,
    HOUGH_DP, HOUGH_MIN_DIST, HOUGH_CANNY_HIGH, HOUGH_ACCUM_THRESH, BALL_HSV_RANGES
    )

class BallDetector:
    def __init__(self, table_px_size, table_m_size, min_px_radius, max_px_radius, green_hsv_range):
        
        self.W, self.H = table_px_size  # Table width and height in pixels.
        self.Wm, self.Hm = table_m_size  # Table width and height in meters.
        self.min_r = min_px_radius  # Minimum ball radius in pixels.
        self.max_r = max_px_radius  # Maximum ball radius in pixels.
        self.green_lower, self.green_upper = green_hsv_range  # HSV range for green color.
        self.ball_hsv_ranges = BALL_HSV_RANGES

        # Precompute scale factors for converting pixel coordinates to meters.
        self.scale_x = self.Wm / self.W  # Scale factor for x-axis.
        self.scale_y = self.Hm / self.H  # Scale factor for y-axis.

        # Morphological kernel for noise removal and contour smoothing.
        # A 5x5 elliptical kernel is used for morphological operations.
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    def detectBalls(self, warped: np.ndarray) -> list[tuple[int,int]]:
        # 1) mask out the green table
        hsv        = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        table_mask = cv.inRange(hsv, self.green_lower, self.green_upper)
        ball_mask  = cv.bitwise_not(table_mask)

        # 2) denoise with your kernel
        k = HOUGH_KERNEL_SIZE
        kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        proc = cv.morphologyEx(ball_mask, cv.MORPH_OPEN, kern, iterations=1)
        proc = cv.morphologyEx(proc,      cv.MORPH_CLOSE, kern, iterations=1)

        # 3) prepare gray/blur for Hough
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (k, k), 0)

        # 4) run HoughCircles with your tuned params
        circles = cv.HoughCircles(
            blur, cv.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DIST,
            param1=HOUGH_CANNY_HIGH,
            param2=HOUGH_ACCUM_THRESH,
            minRadius=HOUGH_MIN_RADIUS,
            maxRadius=HOUGH_MAX_RADIUS
        )

        centers = []
        if circles is not None:
            for x, y, r in np.round(circles[0]).astype(int):
                centers.append((int(x), int(y)))

        return centers

    def detectColoredBalls(self, warped: np.ndarray) -> list[dict]:
        hsv       = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        table_mask = cv.inRange(hsv, self.green_lower, self.green_upper)
        ball_area  = cv.bitwise_not(table_mask)

        positions: dict[str, list[tuple[int,int]]] = {}

        for color_name, (lower, upper) in self.ball_hsv_ranges.items():
            # isolate just this color
            color_mask = cv.inRange(hsv, lower, upper)
            mask = cv.bitwise_and(ball_area, ball_area, mask=color_mask)

            # clean up noise
            kernel = np.ones((5,5), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            # find the blobs
            contours, _ = cv.findContours(mask,
                                        cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                (x, y), radius = cv.minEnclosingCircle(cnt)
                if self.min_r < radius < self.max_r:
                    positions.setdefault(color_name, []).append(
                        (int(x), int(y))
                    )
        return positions