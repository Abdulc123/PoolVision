import cv2 as cv
import numpy as np
from camera.capture import CameraCapture
from camera.tableWarper import get_warped_frame, ARUCO_DICT, ARUCO_PARAMS, TABLE_W, TABLE_H
from cv2 import aruco

def main():
    camera = CameraCapture(camera_index=1, width=1280, height=720)
    params = aruco.DetectorParameters()

    # Adaptive‐Thresh window sizes: smaller windows = more local adaptivity  
    params.adaptiveThreshWinSizeMin    = 3  
    params.adaptiveThreshWinSizeMax    = 23  
    params.adaptiveThreshWinSizeStep   = 10  

    # Constant subtracted from mean to fine-tune binarization  
    params.adaptiveThreshConstant      = 7  

    # Ignore tiny/huge blobs  
    params.minMarkerPerimeterRate      = 0.03  
    params.maxMarkerPerimeterRate      = 4.0  

    # Approximation accuracy (lower = tighter polygon fit)  
    params.polygonalApproxAccuracyRate = 0.05  

    # Enable sub-pixel corner refinement for sharper corner localization  
    params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX  
    params.cornerRefinementWinSize     = 5  
    params.cornerRefinementMaxIterations = 30  
    params.cornerRefinementMinAccuracy = 0.1  

    while True:
        frame = camera.get_frame()

        # 1) Gaussian blur to smooth out sensor noise
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 1) Gaussian blur to smooth out sensor noise
        gray = cv.GaussianBlur(gray, (5,5), 0)

        # 2) Histogram equalization to boost contrast
        gray = cv.equalizeHist(gray)
        
        # detect
        corners, ids, rejected = aruco.detectMarkers(gray, ARUCO_DICT, parameters=params)

        # debug view
        debug = frame.copy()
        if ids is not None:
            aruco.drawDetectedMarkers(debug, corners, ids)
        cv.imshow("Preprocessed (EQ)", gray)
        cv.imshow("Marker Debug", debug)    

        # 3) Show the debug window
        cv.imshow("Marker Debug", debug)

        # 4) Try to warp
        warped = get_warped_frame(frame)
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