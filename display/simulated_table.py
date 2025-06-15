import cv2 as cv
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

    def getBallPositionsFrame(self, model_results, reference_frame):
        
        """
        Draws balls on a new blank frame using the color map and class names.
        Stripes are drawn as circles with two vertical lines.
        """
        self.model_results = model_results 
        self.colored_ball_positions = self.ballDetector.getBallColorPositionsFromModel(self.model_results)
        return self.drawBallPositionsFrame(reference_frame)
    
    def drawBallPositionsFrame(self, reference_frame):
        # Create a blank frame with the same shape as the reference frame
        new_frame = np.zeros_like(reference_frame)
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

        return new_frame