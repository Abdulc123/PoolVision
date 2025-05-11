import cv2 as cv 
import numpy as np

def detectTableEdges(frame):
    """
    Detect table edges and return a warped_frame tob-down view of the pool table
    """
    # Resize the frame for performance
    resized_frame = cv.resize(frame, (640,360))

    # Convert the image to greyscale for edge detection
    gray_image = cv.cvtColor(resized_frame, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smooth the image
    blurred_image = cv.GaussianBlur(gray_image, (5,5), 0) 
    # Kernel Size (Square window used to average out and blur image) Changes blur intensity

    # Detect edges using the Canny edge detector
    edges = cv.Canny(blurred_image, 50, 150)
    # 50 = lower threshold, 150 = higher_threshold for edge detection (Allows edges with value above the higher threshhold, and those edges connected lower to higherthreshholds)
    
    # Find the contours from the edge detected image
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Contour is the boundary between the white pixel and the black pixel
    # RETR_EXTERNAL: Only retrieved external contours (parent contour)
    # CHAIN_APPROX_SIMPLE: Removes redundant poitns to compress contour representation

    # Assume the largest contour (parent contour) is the pool table based off of RETR_EXTERNAL
    largest_countour = max(contours, key=cv.contourArea)

    # Approximate the countour to a polygon
    epsilon = 0.02 * cv.arcLength(largest_countour, True) # 2% of the countour perimeter
    approx = cv.approxPolyDP(largest_countour, epsilon, True) 
    # Simplifies contour shape to fewer points

    # If 4 points are detected, assume its a rectangle (the pool table)
    if len(approx) == 4:
        # Extract the points and reorder them
        points = approx.reshape(4,2)
        ordered_points = orderPoints(points)
    
        # Define destination points for the top down view (same aspect ratio)
        destination_points = np.array([
            [0,0],
            [640, 0],
            [640, 360],
            [0, 360]
        ], dtype="float32")

        # Compute the perspective transform matrix and apply it
        matrix = cv.getPerspectiveTransform(ordered_points, destination_points)
        warped_frame = cv.warpPerspective(resized_frame, matrix, (640, 360))

        return warped_frame, ordered_points
    else:
        print("Detector: [Warning] Table shape could not be approximated to 4 corners")
        return resized_frame, None


def orderPoints(points):
    """
    Orders 4 points in the following order: top-left, top_right, bottom_right, bottom_left
    """
    rectangle = np.zeros((4,2), dtype="float32")

    # Sum and diff of points helps us find the corners
    s = points.sum(axis = 1) # Top left will have the smallest sum
    diff = np.diff(points, axis = 1) # Top right will have the smallest difference

    rectangle[0] = points[np.argmin(s)]     # Top Left
    rectangle[1] = points[np.argmax(s)]     # Bottom Right
    rectangle[2] = points[np.argmin(diff)]  # Top Right
    rectangle[3] = points[np.argmax(diff)]  # Bottom Left

    return rectangle
