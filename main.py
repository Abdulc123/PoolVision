import cv2 as cv
import numpy as np
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame, applyGrayFiltersToFrameAndDetectMarkers, TABLE_W, TABLE_H
from cv2 import aruco

def main():
    camera = CameraCapture(camera_index=1, width=1280, height=720)
    
    while True:
        frame = camera.get_frame()        
        warped = getWarpedFrame(frame, debug_mode=True)
        
        if warped is not None:
            cv.imshow("Warped Table", warped)
        else:
            # optional: show a blank so you know warp failed
            cv.imshow("Warped Table", np.zeros((TABLE_H, TABLE_W, 3), dtype=np.uint8))

        # 5) quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()