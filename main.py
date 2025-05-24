import cv2 as cv
import numpy as np
import config.settings as Settings
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame, saveCache
from processing.BallDetector import BallDetector
from display.screenDisplay import drawBallPositions

def main():
    camera = CameraCapture(camera_index=0, width=Settings.WARPED_TABLE_W, height=Settings.WARPED_TABLE_H)
    Settings.detector = BallDetector(
        table_px_size   = (Settings.WARPED_TABLE_W, Settings.WARPED_TABLE_H),
        table_m_size    = (Settings.TABLE_WIDTH_M, Settings.TABLE_HEIGHT_M),
        min_px_radius   = Settings.MIN_BALL_RADIUS_PX,
        max_px_radius   = Settings.MAX_BALL_RADIUS_PX,
        green_hsv_range = (Settings.LOWER_GREEN_HSV_RANGE,
                           Settings.UPPER_GREEN_HSV_RANGE)
    )

    try:       
        while True:
            frame = camera.get_frame()  
            warped = getWarpedFrame(frame, debug_mode=False)  
            ball_positions = Settings.detector.detectBalls(warped)
            drawBallPositions(ball_positions, warped)

            # print(ball_positions)

            if warped is not None:
                hsv   = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
                mask  = cv.inRange(hsv, Settings.LOWER_GREEN_HSV_RANGE, Settings.UPPER_GREEN_HSV_RANGE)

                cv.imshow("Warped Table", warped)
                cv.imshow("Masked Warped Table",  mask)
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