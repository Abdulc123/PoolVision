import cv2 as cv
import numpy as np
import config.settings as Settings
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame, saveCache


def main():
    camera = CameraCapture(camera_index=1, width=1280, height=720)
    
    try:       
        while True:
            frame = camera.get_frame()        
            warped = getWarpedFrame(frame, debug_mode=True)

            if warped is not None:
                cv.imshow("Warped Table", warped)
            else:
                # optional: show a blank so you know warp failed
                cv.imshow("Warped Table", np.zeros((Settings.WARPED_TABLE_H, Settings.WARPED_TABLE_W, 3), dtype=np.uint8))

            # 5) quit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    finally:
        saveCache()
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()