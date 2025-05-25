import cv2 as cv
import numpy as np

from config.settings import (
    HOUGH_KERNEL_SIZE, HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS,
    HOUGH_DP, HOUGH_MIN_DIST, HOUGH_CANNY_HIGH, HOUGH_ACCUM_THRESH, BALL_HSV_RANGES, BALL_HUE_RANGES,
    WHITE_SAT_THRESH, WHITE_VAL_THRESH, STRIPE_FRAC_THRESH, SAMPLE_COLOR_RADIUS
    )

class BallDetector:
    def __init__(self, table_px_size, table_m_size, min_px_radius, max_px_radius, green_hsv_range, alpha = 0.3):

        self.W, self.H = table_px_size  # Table width and height in pixels.
        pocket_locations = np.load('config/pockets.npy')
        self.pocket_mask = np.zeros((self.H,     self.W), dtype=np.uint8)
        for (x, y, pocket_radius) in pocket_locations:
            cv.circle(self.pocket_mask, (int(x),int(y)), pocket_radius, 255, -1)

        self.Wm, self.Hm = table_m_size  # Table width and height in meters.
        self.min_r = min_px_radius  # Minimum ball radius in pixels.
        self.max_r = max_px_radius  # Maximum ball radius in pixels.
        self.green_lower, self.green_upper = green_hsv_range  # HSV range for green color.
        self.ball_hsv_ranges = BALL_HSV_RANGES
        self.ball_hue_ranges = BALL_HUE_RANGES

        # Precompute scale factors for converting pixel coordinates to meters.
        self.scale_x = self.Wm / self.W  # Scale factor for x-axis.
        self.scale_y = self.Hm / self.H  # Scale factor for y-axis.

        # Morphological kernel for noise removal and contour smoothing.
        # A 5x5 elliptical kernel is used for morphological operations.
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        # self.multi = cv.legacy.MultiTracker_create()
        # self.initialized = False

        self.colored_ball_positions = {} # "Ball Color" : (x,y)

    # def reInitialize(self, frame, ball_positions):
    #     # ball_positions: list of (x,y)
    #     self.multi = cv.legacy.MultiTracker_create()
    #     for x,y in ball_positions:
    #         size = 30
    #         bbox = (x - size//2, y - size//2, size, size)
    #         self.multi.add(cv.legacy.TrackerCSRT_create(), frame, bbox)
    #     self.initialized = True

    # def update(self, frame):
    #     if not self.initialized:
    #         return []
    #     boxes = self.multi.getObjects()
    #     centers = []
    #     for box in boxes:
    #             x,y,w,h = box
    #             centers.append((int(x + w/2), int(y + h/2)))
    #     return centers
    
    def rawDetectBalls(self, warped: np.ndarray) -> list[tuple[int,int]]:
        # 1) mask out the green table
        hsv        = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        table_mask = cv.inRange(hsv, self.green_lower, self.green_upper)
        ball_mask  = cv.bitwise_not(table_mask)
        ball_mask  = cv.bitwise_and(ball_mask, cv.bitwise_not(self.pocket_mask))

        # 2) denoise with your kernel
        k = HOUGH_KERNEL_SIZE
        kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        proc = cv.morphologyEx(ball_mask, cv.MORPH_OPEN, kern, iterations=1)
        proc = cv.morphologyEx(proc,      cv.MORPH_CLOSE, kern, iterations=1)
        proc = cv.morphologyEx(ball_mask, cv.MORPH_CLOSE, kern, iterations=2)


        # 3) prepare gray/blur for Hough
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        masked = cv.bitwise_and(gray, gray, mask=proc)
        blur = cv.GaussianBlur(masked, (k, k), 0)

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
    
    def getBallColorPositions(self, warped, ball_positions, radius = SAMPLE_COLOR_RADIUS):
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        self.colored_ball_positions.clear()

        for x,y in ball_positions:
            # Circle mask
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv.circle(mask, (x,y), radius, 255, -1)

            # Sample pixels for getting color of ball
            pixels = hsv[mask==255] 
            H, S, V = pixels[:,0], pixels[:,1], pixels[:,2]

            # Detect the white ish areas
            is_white = (S < WHITE_SAT_THRESH) & (V > WHITE_VAL_THRESH)
            percentage_white =  is_white.sum() / len(pixels)
            # print(f"[DEBUG] S.mean={S.mean():.1f}, V.mean={V.mean():.1f}  at center {(x,y)}")

            # Determine if its the cue ball
            if percentage_white > 0.9:
                color, ball_type = 'white', 'solid'

            # Determine if its the 8-ball
            elif (S.mean() < 120) and (V.mean() < 120):
                color, ball_type = 'black', 'solid'

            else:
                # Determine stripe vs solid for the ball
                ball_type = 'stripe' if percentage_white >= STRIPE_FRAC_THRESH else 'solid'

                # Histogram on the non white pixels
                color_hues = H[~is_white]
                hist = cv.calcHist([color_hues],[0],None,[180],[0,180]).ravel()
                peak = int(np.argmax(hist))

                # Lookup the calculated hue to see if its within my ranges
                for clr, hue_pair in self.ball_hue_ranges.items():
                    if hue_pair is None:
                        continue

                    h_lo, h_hi = hue_pair
                    # wrap-around for red/maroon if needed
                    if h_lo <= h_hi:
                        if h_lo <= peak <= h_hi:
                            color = clr
                            break
                    else:
                        # hi < lo means wrap (e.g. 170→10)
                        if peak >= h_lo or peak <= h_hi:
                            color = clr
                            break
                else:
                    color = 'unknown'

            key = f"{color}_{ball_type}"
            self.colored_ball_positions[key] = (x, y)
        
        return self.colored_ball_positions
                    