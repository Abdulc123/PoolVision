import cv2 as cv
import json
import os
import numpy as np
from processing.BallDetector import initializeBallDetector
import config.settings as Settings

class SimulatedTable:

    def __init__(self):
        self.colored_ball_positions = None
        self.model_results = None
        self.radius = 20
        self.color_map = Settings.COLOR_MAP
        self.ballDetector = initializeBallDetector()
        self.cushion_points = loadCushionPoints()
        self.pocket_positions = loadPocketPositions()

    
    def getSimulatedTableFrame(self, model_results, reference_frame):
        # Create a blank frame with the same shape as the reference frame

        self.model_results = model_results 
        self.colored_ball_positions = self.ballDetector.getBallColorPositionsFromModel(self.model_results)

        self.drawCushionBorders(reference_frame, self.cushion_points)
        # self.drawBallPositions(reference_frame)
        
        return reference_frame
    
    def drawBallPositions(self, new_frame):
        for class_name, positions in self.colored_ball_positions.items():
            color = self.color_map.get(class_name, (255, 255, 255))
            for (x, y) in positions:
                if 'stripe' in class_name:
                    # Draw circle
                    cv.circle(new_frame, (x, y), self.radius, color, 2)
                    # Draw two vertical lines
                    cv.line(new_frame, (x - self.radius//2, y - self.radius), (x - self.radius//2, y + self.radius), color, 2)
                    cv.line(new_frame, (x + self.radius//2, y - self.radius), (x + self.radius//2, y + self.radius), color, 2)
                else:
                    # Draw filled circle for solids/cue/8-ball
                    cv.circle(new_frame, (x, y), self.radius, color, -1)
    
    def drawCushionBorders(self, frame, cushion_points):
        # Draw the cushion rectangle
        pts = np.array(cushion_points, np.int32).reshape((-1, 1, 2))
        cv.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
        # Optionally, label the corners
        for idx, (x, y) in enumerate(cushion_points):
            cv.putText(frame, f"{idx}", (x+10, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def loadCushionPoints():
    cushion_file = os.path.join(os.path.dirname(__file__), "../calibrators/JsonData/calibrateCushions.json")
    with open(cushion_file, "r") as f:
        cushion_cache = json.load(f)
    # Convert to list of tuples in order: TL, TR, BR, BL
    points = [tuple(map(int, cushion_cache[str(i)])) for i in range(4)]
    return points

def loadPocketPositions():
    # Load pocket positions from JSON
    pockets_path = os.path.join(os.path.dirname(__file__), "../calibrators/JsonData/pockets.json")
    with open(pockets_path, "r") as f:
        pockets_dict = json.load(f)
    # Ensure order: 0=TL, 1=TM, 2=TR, 3=BR, 4=BM, 5=BL
    pocket_positions = [tuple(pockets_dict[str(i)]) for i in range(6)]
    return pocket_positions