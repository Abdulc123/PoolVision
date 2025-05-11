import cv2 as cv
import numpy as np

# Global variables
clicked_hsv = None
frame_hsv = None

# Tolerance for generating HSV range
H_TOLERANCE = 15
S_TOLERANCE = 60
V_TOLERANCE = 60

def mouse_callback(event, x, y, flags, param):
    global clicked_hsv, frame_hsv

    if event == cv.EVENT_LBUTTONDOWN:
        hsv_val = frame_hsv[y, x]
        clicked_hsv = hsv_val

        h, s, v = hsv_val
        lower = (
            max(h - H_TOLERANCE, 0),
            max(s - S_TOLERANCE, 0),
            max(v - V_TOLERANCE, 0)
        )
        upper = (
            min(h + H_TOLERANCE, 179),
            min(s + S_TOLERANCE, 255),
            min(v + V_TOLERANCE, 255)
        )

        print(f"[INFO] Clicked HSV: {tuple(hsv_val)}")
        print(f"[INFO] Suggested HSV Range:")
        print(f"HSV_LOWER_GREEN = {lower}")
        print(f"HSV_UPPER_GREEN = {upper}")

def main():
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)  # Use your external cam index

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    cv.namedWindow("Live HSV Picker")
    cv.setMouseCallback("Live HSV Picker", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        global frame_hsv
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        cv.imshow("Live HSV Picker", frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    # This script allows you to click on a pixel in the camera feed and get the HSV value of that pixel.