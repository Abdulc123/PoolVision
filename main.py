import cv2 as cv
import numpy as np
from camera.capture import CameraCapture
from processing.detector import isolateTable

def main():
    camera = CameraCapture(camera_index=1, width=1280, height=720)

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("[ERROR] Failed to capture frame.")
            break

        warped, mask = isolateTable(frame)

        if warped is not None:
            cv.imshow("Warped Table View", warped)
        else:
            print("[INFO] Warped view could not be generated.")
        cv.imshow("Green Mask", mask)

        cv.imshow("Raw View", frame)
        cv.imshow("Green Mask", mask)

        key = cv.waitKey(1)
        if key == ord('q') or cv.getWindowProperty("Raw View", cv.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()