import cv2 as cv
import numpy as np
from config.settings import WARPED_TABLE_W, WARPED_TABLE_H

# map your ball keys to BGR colors:
COLOR_MAP = {
    'white_ball': (255, 255, 255),
    'blue_ball':  (255,   0,   0),
    'red_ball':   (  0,   0, 255),
    'purple_ball': (128,   0, 128),
    'yellow_ball': (  0, 255, 255),
    'orange_ball': (  0, 165, 255),
    'maroon_ball': (153,   0,   0),
    'green_ball':  (  0, 255,   0),
    'black_ball':  (  0,   0,   0),
}

def drawColoredBallPositions(positions: dict[str, list[tuple[int,int]]], warped_frame):
    """
    Draws the balls on a black canvas and shows them in a window.
    """
    # canvas = np.zeros((WARPED_TABLE_H, WARPED_TABLE_W, 3), dtype=np.uint8)
    cv.rectangle(warped_frame, (0,0), (WARPED_TABLE_W-1, WARPED_TABLE_H-1),
                 (255,255,200), 2)

    for name, pts in positions.items():
        color = COLOR_MAP.get(name, (0,255,255))
        for (x,y) in pts:
            cv.circle(warped_frame, (x,y), 20, color, 3)

    cv.imshow("Ball Positions", warped_frame)

def drawBallPositions(positions: list[tuple[int,int]], warped_frame):
    """
    Draws the balls on the warped table view and displays them in a window
    """
    for x,y in positions:
        cv.circle(warped_frame, (x,y), 20, (255,0,0), 3)
            