import cv2 as cv
import numpy as np
import config.settings as Settings


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

        # Precompute scale factors for converting pixel coordinates to meters.
        self.scale_x = self.Wm / self.W  # Scale factor for x-axis.
        self.scale_y = self.Hm / self.H  # Scale factor for y-axis.

        self.colored_ball_positions = {} # "Ball Color" : (x,y)

    def getBallColorPositionsFromModel(self, results, color_map, radius=20):
        colored_ball_positions = {}
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = r.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if class_name not in colored_ball_positions:
                    colored_ball_positions[class_name] = []
                colored_ball_positions[class_name].append((center_x, center_y))
        return colored_ball_positions

def drawColoredBallPositions(colored_ball_positions, frame, color_map, radius=20):
    """
    Draws balls on the frame using the color map and class names.
    Stripes are drawn as circles with two vertical lines.
    """
    for class_name, positions in colored_ball_positions.items():
        color = color_map.get(class_name, (255, 255, 255))
        for (x, y) in positions:
            if 'stripe' in class_name:
                # Draw circle
                cv.circle(frame, (x, y), radius, color, 2)
                # Draw two vertical lines
                cv.line(frame, (x - radius//2, y - radius), (x - radius//2, y + radius), color, 2)
                cv.line(frame, (x + radius//2, y - radius), (x + radius//2, y + radius), color, 2)
            else:
                # Draw filled circle for solids/cue/8-ball
                cv.circle(frame, (x, y), radius, color, -1)

def drawColoredBallPositionsOnNewFrame(colored_ball_positions, reference_frame, color_map, radius=20):
    """
    Draws balls on a new blank frame using the color map and class names.
    Stripes are drawn as circles with two vertical lines.
    """
    # Create a blank frame with the same shape as the reference frame
    new_frame = np.zeros_like(reference_frame)

    for class_name, positions in colored_ball_positions.items():
        color = color_map.get(class_name, (255, 255, 255))
        for (x, y) in positions:
            if 'stripe' in class_name:
                # Draw circle
                cv.circle(new_frame, (x, y), radius, color, 2)
                # Draw two vertical lines
                cv.line(new_frame, (x - radius//2, y - radius), (x - radius//2, y + radius), color, 2)
                cv.line(new_frame, (x + radius//2, y - radius), (x + radius//2, y + radius), color, 2)
            else:
                # Draw filled circle for solids/cue/8-ball
                cv.circle(new_frame, (x, y), radius, color, -1)
    return new_frame

def initializeBallDetector():
    ballDetector = BallDetector(
        table_px_size=(Settings.WARPED_TABLE_W, Settings.WARPED_TABLE_H),
        table_m_size=(Settings.TABLE_WIDTH_M, Settings.TABLE_HEIGHT_M),
        min_px_radius=Settings.MIN_BALL_RADIUS_PX,
        max_px_radius=Settings.MAX_BALL_RADIUS_PX,
        green_hsv_range=(Settings.LOWER_GREEN_HSV_RANGE, Settings.UPPER_GREEN_HSV_RANGE)
    )

    return ballDetector