import cv2 as cv
import numpy as np
from config.settings import WARPED_TABLE_W, WARPED_TABLE_H, BALL_RADIUS, COLOR_MAP_SOLIDS, COLOR_MAP_STRIPES



def drawColoredBallPositions(colored_ball_positions: dict[str, tuple[int,int]], warped_frame):
    """
    Draws the balls on a black canvas and shows them in a window.
    """
    # canvas = np.zeros((WARPED_TABLE_H, WARPED_TABLE_W, 3), dtype=np.uint8)
    cv.rectangle(warped_frame, (0,0), (WARPED_TABLE_W-1, WARPED_TABLE_H-1), (255,255,200), 2)

    for name, position in colored_ball_positions.items():
        x,y = position
        color = (255,0,255)
        if name in COLOR_MAP_SOLIDS.keys():
            color = COLOR_MAP_SOLIDS.get(name, (0,255,255))

        elif name in COLOR_MAP_STRIPES.keys():
            color = COLOR_MAP_STRIPES.get(name, (0,255,255))
            # Draw vertical black lines on the left and right sides of the circle
            cv.line(warped_frame, (x - BALL_RADIUS // 2, y - BALL_RADIUS), (x - BALL_RADIUS // 2, y + BALL_RADIUS), (0, 0, 0), 2)
            cv.line(warped_frame, (x + BALL_RADIUS // 2, y - BALL_RADIUS), (x + BALL_RADIUS // 2, y + BALL_RADIUS), (0, 0, 0), 2)

        cv.circle(warped_frame, (x,y), BALL_RADIUS, color, 3)

    cv.imshow("Ball Positions", warped_frame)

def drawBallPositions(positions: list[tuple[int,int]], warped_frame, simulatedTable=False):
    """
    Draws the balls on the warped table view and displays them in a window
    """
    frame = warped_frame
    if simulatedTable:
        canvas = np.zeros((WARPED_TABLE_H, WARPED_TABLE_W, 3), dtype=np.uint8)
        cv.rectangle(canvas, (0,0), (WARPED_TABLE_W-1, WARPED_TABLE_H-1), (255,255,200), 2)
        frame = canvas
        for x,y in positions:
            cv.circle(frame, (x,y), BALL_RADIUS, (255,0,0), 3)
            cv.circle(frame, (x, y), 2, (0, 0, 255), 3)
        cv.imshow("Simulated Table Positions", frame)
    else:
        for x,y in positions:
            cv.circle(frame, (x,y), BALL_RADIUS, (255,0,0), 3)
            cv.circle(frame, (x, y), 2, (0, 0, 255), 3)
                