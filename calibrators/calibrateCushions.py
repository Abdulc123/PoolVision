import cv2 as cv
import numpy as np
import json
import os
from camera.capture import CameraCapture
import config.settings as Settings

# python -m calibrators.calibrateWarpedTable

# Load existing corner cache
_CACHE_FILE = os.path.join(os.path.dirname(__file__), "calibrateCushions.json")
if os.path.exists(_CACHE_FILE):
    with open(_CACHE_FILE, "r") as f:
        corner_cache = json.load(f)
    corner_cache = {int(k): np.array(v, dtype=np.float32) for k, v in corner_cache.items()}
else:
    corner_cache = {0: [0,0], 1: [0,0], 2: [0,0], 3: [0,0]}  # fallback default

# Convert to editable list format
points = [corner_cache[i] for i in range(4)]
points = [tuple(map(int, p)) for p in points]

dragging_point = None

def mouse_callback(event, x, y, flags, param):
    global dragging_point
    if event == cv.EVENT_LBUTTONDOWN:
        # Check if clicking near a point
        for i, (px, py) in enumerate(points):
            if abs(px - x) < 15 and abs(py - y) < 15:
                dragging_point = i
    elif event == cv.EVENT_MOUSEMOVE and dragging_point is not None:
        # Update dragged point
        points[dragging_point] = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        dragging_point = None

def main():
    global points

    camera = CameraCapture(camera_index=0, width=Settings.WARPED_TABLE_W, height=Settings.WARPED_TABLE_H)

    cv.namedWindow("Calibrator")
    cv.setMouseCallback("Calibrator", mouse_callback)

    print("Drag the points to adjust corners.")
    print("Press 's' to save to corner_cache.json")
    print("Press 'q' to quit without saving")

    while True:
        frame = camera.get_frame()

        # Draw current points + connecting rectangle
        for idx, p in enumerate(points):
            cv.circle(frame, p, 10, (0, 255, 0), -1)
            cv.putText(
                frame,
                f"{idx}: {['TL','TR','BR','BL'][idx]}",
                (p[0]+12, p[1]-12),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        cv.polylines(frame, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

        cv.imshow("Calibrator", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save adjusted points
            new_cache = {str(i): [float(p[0]), float(p[1])] for i, p in enumerate(points)}
            with open(_CACHE_FILE, "w") as f:
                json.dump(new_cache, f, indent=2)
            print(f"Saved new points to {_CACHE_FILE}")
            break

        if key == ord('q'):
            print("Exiting without saving.")
            break

    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
