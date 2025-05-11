import cv2 as cv
import numpy as np
from config.settings import HSV_LOWER_GREEN, HSV_UPPER_GREEN


def isolateTable(frame):
    """
    Isolates the pool table from the given frame and returns a top-down warped view of the table.
    
    Parameters:
    - frame: The input image (numpy array) in BGR format from which the pool table needs to be isolated.

    Returns:
    - warped: The top-down warped view of the pool table (numpy array). Returns None if the table cannot be isolated.
    - mask: The binary mask used to detect the pool table based on the HSV color range.
    """
    # Convert the input frame from BGR to HSV color space for easier color segmentation
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Create a binary mask where the green color of the pool table is within the specified HSV range
    mask = cv.inRange(hsv, np.array(HSV_LOWER_GREEN), np.array(HSV_UPPER_GREEN))

    # Find contours in the binary mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.RETR_EXTERNAL: Retrieves only the outermost contours
    # cv.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments to save memory

    # If no contours are found, return None for the warped image and the mask
    if not contours:
        return None, mask

    # Find the largest contour by area, assuming it corresponds to the pool table
    largest = max(contours, key=cv.contourArea)

    # Approximate the contour to a polygon with fewer vertices
    epsilon = 0.02 * cv.arcLength(largest, True)  # 2% of the contour's perimeter
    approx = cv.approxPolyDP(largest, epsilon, True)

    # If the approximated polygon does not have 4 vertices, return None and the mask
    if len(approx) != 4:
        return None, mask

    # Reshape the 4 vertices into a 2D array for further processing
    pts = approx.reshape(4, 2)

    # Initialize an empty array to store the ordered rectangle points
    rect = np.zeros((4, 2), dtype="float32")

    # Order the points: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)  # Sum of x and y coordinates
    rect[0] = pts[np.argmin(s)]  # Top-left has the smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has the largest sum

    diff = np.diff(pts, axis=1)  # Difference between x and y coordinates
    rect[1] = pts[np.argmin(diff)]  # Top-right has the smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has the largest difference

    rect = expandRectanglePoints(rect, 40)  # Expand the rectangle points by 10 pixels

    # Define the dimensions of the output warped image
    (w, h) = (720, 400)  # Width and height of the warped image

    # Define the destination points for the perspective transform
    dst = np.array([
        [0, 0],          # Top-left corner
        [w - 1, 0],      # Top-right corner
        [w - 1, h - 1],  # Bottom-right corner
        [0, h - 1]       # Bottom-left corner
    ], dtype="float32")

    # Compute the perspective transform matrix
    M = cv.getPerspectiveTransform(rect, dst)

    # Apply the perspective transform to get a top-down view of the pool table
    warped = cv.warpPerspective(frame, M, (w, h))

    # Return the warped image and the binary mask
    return warped, mask

def expandRectanglePoints(rect, expand_px):
    """
    Expands the rectangle points outward by a specified number of pixels.

    Parameters:
    - rect: The rectangle points (numpy array) to be expanded.
    - expand_px: The number of pixels to expand the rectangle.

    Returns:
    - expanded_rect: The expanded rectangle points (numpy array).
    """
    expanded_rect = np.copy(rect)
    
    # Top-left: move left and up
    expanded_rect[0][0] -= expand_px
    expanded_rect[0][1] -= expand_px

    # Top-right: move right and up
    expanded_rect[1][0] += expand_px
    expanded_rect[1][1] -= expand_px

    # Bottom-right: move right and down
    expanded_rect[2][0] += expand_px
    expanded_rect[2][1] += expand_px

    # Bottom-left: move left and down
    expanded_rect[3][0] -= expand_px
    expanded_rect[3][1] += expand_px

    return expanded_rect