import cv2 as cv
import json
import os
import numpy as np
from processing.BallDetector import initializeBallDetector
import config.settings as Settings

def getCueDirectionFromModel(model_results):
    for box in model_results:
        if box['name'] == 'cue_body':
            x1, y1, x2, y2 = map(int, box['box'])  # top-left and bottom-right
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dx = x2 - x1
            dy = y2 - y1
            direction = np.array([dx, dy])
            norm = np.linalg.norm(direction)
            if norm == 0:
                return None
            direction = direction / norm
            return (cx, cy), direction
    return None, None

def drawBouncingLine(frame, start_pos, direction, cushion_lines, bounces=3):
    pos = np.array(start_pos, dtype=np.float32)
    dir_vec = np.array(direction, dtype=np.float32)

    for _ in range(bounces):
        intersection, normal = getIntersectionAndNormal(pos, dir_vec, cushion_lines)
        if intersection is None:
            break
        intersection = np.array(intersection)
        cv.line(frame, tuple(pos.astype(int)), tuple(intersection.astype(int)), (255, 255, 255), 2)
        # Reflect
        dir_vec = dir_vec - 2 * np.dot(dir_vec, normal) * normal
        pos = intersection

def getCushionLines(cushion_points):
    return [
        (cushion_points[i], cushion_points[(i + 1) % 4])
        for i in range(4)
    ]


def loadCushionPositions():
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

def getIntersectionAndNormal(start_point, direction, cushion_lines):
    """
    Finds the first intersection of a line with any cushion line.
    Returns the intersection point and the normal vector of the cushion.
    """
    min_dist = float('inf')
    closest_intersection = None
    closest_normal = None

    for (p1, p2) in cushion_lines:
        intersection = lineIntersection(start_point, direction, p1, p2)
        if intersection is not None:
            dist = np.linalg.norm(np.array(intersection) - np.array(start_point))
            if dist < min_dist:
                min_dist = dist
                closest_intersection = intersection
                # Calculate cushion normal (perpendicular to the wall)
                edge = np.array(p2) - np.array(p1)
                normal = np.array([-edge[1], edge[0]])  # 90° rotation
                normal = normal / np.linalg.norm(normal)
                # Flip normal if facing same direction as incoming vector
                if np.dot(direction, normal) > 0:
                    normal = -normal
                closest_normal = normal

    return closest_intersection, closest_normal

def lineIntersection(ray_origin, ray_direction, seg_a, seg_b):
    """
    Returns intersection point between a ray (origin + direction)
    and a line segment (seg_a to seg_b), or None if no intersection.
    """
    p = np.array(ray_origin, dtype=np.float32)
    r = np.array(ray_direction, dtype=np.float32)
    q = np.array(seg_a, dtype=np.float32)
    s = np.array(seg_b, dtype=np.float32) - q

    r_cross_s = np.cross(r, s)
    if r_cross_s == 0:
        return None  # Parallel lines

    t = np.cross((q - p), s) / r_cross_s
    u = np.cross((q - p), r) / r_cross_s

    if t >= 0 and 0 <= u <= 1:
        intersection = p + t * r
        return tuple(intersection.astype(int))
    return None

def getBallColorPositionsFromProcessedResults(model_results):
    """
    Returns a dict: {class_name: [(center_x, center_y), ...]}
    """
    colored_ball_positions = {}
    for box in model_results:
        class_name = box['name']
        x1, y1, x2, y2 = box['box']
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if class_name not in colored_ball_positions:
            colored_ball_positions[class_name] = []
        colored_ball_positions[class_name].append((center_x, center_y))
    return colored_ball_positions

class SimulatedTable:
    def __init__(self):
        self.colored_ball_positions = None
        self.model_results = None
        self.radius = 20
        self.color_map = Settings.COLOR_MAP
        self.cushion_points = loadCushionPositions()

    def getSimulatedTableFrame(self, model_results, reference_frame):
        self.model_results = model_results 
        self.colored_ball_positions = getBallColorPositionsFromProcessedResults(self.model_results)
        self.drawCushionBorders(reference_frame, self.cushion_points)
        cue_position, cue_direction = getCueDirectionFromModel(model_results)
        if cue_position is not None:
            cushion_lines = getCushionLines(self.cushion_points)
            drawBouncingLine(reference_frame, cue_position, cue_direction, cushion_lines, bounces=3) 
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
