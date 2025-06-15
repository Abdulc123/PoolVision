import numpy as np
import cv2 as cv
import threading

# Video and Audio Capture variables
ALLOW_TABLE_RECORDING = True
LISTEN_FOR_AUDIO_COMMANDS = False
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

COLOR_MAP = {
    'black_8ball': (0, 0, 0),
    'blue_solid': (255, 0, 0),
    'blue_stripe': (255, 0, 0),
    'cue_ball': (255, 255, 255),
    'cue_body': (200, 200, 200),   # Example, adjust as needed
    'cue_tip': (50, 50, 50),       # Example, adjust as needed
    'green_solid': (0, 255, 0),
    'green_stripe': (0, 255, 0),
    'maroon_solid': (153, 0, 0),
    'maroon_stripe': (153, 0, 0),
    'orange_solid': (0, 165, 255),
    'orange_stripe': (0, 165, 255),
    'player_body': (100, 100, 100), # Example, adjust as needed
    'player_hand': (200, 180, 120), # Example, adjust as needed
    'playr': (100, 100, 100),       # Example, adjust as needed
    'purple_solid': (128, 0, 128),
    'purple_stripe': (128, 0, 128),
    'red_solid': (0, 0, 255),
    'red_stripe': (0, 0, 255),
    'yellow_solid': (0, 255, 255),
    'yellow_stripe': (0, 255, 255),
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
