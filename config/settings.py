import numpy as np
import cv2 as cv
import threading

# Video and Audio Capture variables
RECORDING_TABLE = True
LISTEN_FOR_AUDIO_COMMANDS = True
TAKE_A_PICTURE_EVENT = threading.Event()

# These dimensions define the size of the output warped image (top-down view of the table).
WARPED_TABLE_W, WARPED_TABLE_H = 1280, 720  # Width and height of the table in pixels

# Width and Height of actual table in Meters: 64.75 inches, 31 inches
TABLE_WIDTH_M, TABLE_HEIGHT_M = 1.64465, 0.7874
pixels_per_m  = WARPED_TABLE_W / TABLE_WIDTH_M
ball_diameter_m = 0.05715 # in meters after warped view
ball_radius_px = (ball_diameter_m * pixels_per_m) / 2

# 20% margin
MIN_BALL_RADIUS_PX = 9 #ball_radius_px  * .8
MAX_BALL_RADIUS_PX = 18 #ball_radius_px * 1.2
BALL_RADIUS = 20

LOWER_GREEN_HSV_RANGE = np.array([40,  52,  99])
UPPER_GREEN_HSV_RANGE = np.array([107,  229,  175])

HOUGH_KERNEL_SIZE   = 7          # Kernel Trackbar
HOUGH_MIN_RADIUS    = 18          # “MinRad”
HOUGH_MAX_RADIUS    = 27        # “MaxRad”
HOUGH_DP            = 15 / 10.0  # “dp*10” → 1.8
HOUGH_MIN_DIST      = 41        # “MinDist”
HOUGH_CANNY_HIGH    = 112        # “CannyHi”
HOUGH_ACCUM_THRESH  = 31         # “AccThresh”

# ighting threshholds
WHITE_SAT_THRESH  =  120   # S < this → “white-ish”
WHITE_VAL_THRESH  = 240   # V > this → “white-ish”
STRIPE_FRAC_THRESH = 0.20  # ≥5% white pixels → stripe
SAMPLE_COLOR_RADIUS = 15

# Ball Color Hue Ranges:
BALL_HSV_RANGES = {
    "white" : (np.array([0, 0, 200]), np.array([179,50, 255])),
    "blue" : (np.array([99,121,102]), np.array([114,255,216])),
    "red"  : (np.array([172,105,192]), np.array([179,181,255])),
    "purple" : (np.array([99,85,68]), np.array([118,205,141])),
    "yellow" : (np.array([20, 150, 150]),np.array([30, 255, 255])),
    "orange" : (np.array([1, 86, 231]), np.array([22, 157, 255])),
    "maroon" : (np.array([151, 34, 80]), np.array([179, 143, 162])),
    "green" : (np.array([86, 196, 64]), np.array([98, 255, 118])),
    "black" : (np.array([0,0,0]), np.array([179, 255,60]))
}

BALL_HUE_RANGES = {
    'blue'  : ( 99, 114),
    'red'   : (172, 179),
    'purple': ( 99, 118),
    'yellow': ( 20,  30),
    'orange': (  1,  22),
    'maroon': (151, 179),
    'green' : ( 86,  98),
    'white' : None, # Cue Ball
    'black' : None  # Eight Ball
}

# Map Ball Keys to Colors:
COLOR_MAP_SOLIDS = {
    'yellow_solid': (  0, 255, 255),
    'blue_solid'  : (255,   0,   0),
    'red_solid'   : (  0,   0, 255),
    'purple_solid': (128,   0, 128),
    'orange_solid': (  0, 165, 255),
    'maroon_solid': (153,   0,   0),
    'green_solid' : (  0, 255,   0),
    'white_solid' : (255, 255, 255),  # cue ball
    'black_solid' : (  0,   0,   0),  # 8-ball
}

COLOR_MAP_STRIPES = {
    'yellow_stripe': (  0, 255, 255),
    'blue_stripe'  : (255,   0,   0),
    'red_stripe'   : (  0,   0, 255),
    'purple_stripe': (128,   0, 128),
    'orange_stripe': (  0, 165, 255),
    'maroon_stripe': (153,   0,   0),
    'green_stripe' : (  0, 255,   0),
}



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
data = np.load('config/cam_calib.npz')
K, dist = data['K'], data['dist']

# 1) Load your one-time calibration results:
data = np.load('config/cam_calib.npz')
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
