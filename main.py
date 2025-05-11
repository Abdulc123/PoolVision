import cv2 as cv
from camera.capture import CameraCapture
from processing.detector import detectTableEdges

def main():
    # 0 for regular camera, 1 for external camera
    camera = CameraCapture(0)
    # camera.show_live_feed()

    while True:
        frame = camera.get_frame()
        warped, corners = detectTableEdges(frame)

        if corners is not None:
            for point in corners:
                cv.circle(warped, tuple(point.astype(int)), 5, (0, 255, 0), -1)

        cv.imshow("Warped Table View", warped)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()