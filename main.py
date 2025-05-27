import cv2 as cv
import time, os, pygame
import config.settings as Settings
import utils.Utils as Utils
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame, saveCache
from processing.BallDetector import BallDetector
from display.screenDisplay import drawBallPositions, drawColoredBallPositions
import pygame

def main():
    Utils.VoiceManager.toggleAudioCommands(print_audio=True)
    camera = CameraCapture(camera_index=0, width=Settings.WARPED_TABLE_W, height=Settings.WARPED_TABLE_H)
    ballDetector = BallDetector(
        table_px_size=(Settings.WARPED_TABLE_W, Settings.WARPED_TABLE_H),
        table_m_size=(Settings.TABLE_WIDTH_M, Settings.TABLE_HEIGHT_M),
        min_px_radius=Settings.MIN_BALL_RADIUS_PX,
        max_px_radius=Settings.MAX_BALL_RADIUS_PX,
        green_hsv_range=(Settings.LOWER_GREEN_HSV_RANGE, Settings.UPPER_GREEN_HSV_RANGE)
    )

    try:
        while Settings.RECORDING_TABLE:
            frame = camera.get_frame()
            if frame is None:
                print("Failed to capture frame.")
                continue

            warped = getWarpedFrame(frame, debug_mode=False)
            if warped is None:
                print("Failed to warp frame.")
                continue

            raw_ball_positions = ballDetector.rawDetectBalls(warped)
            colored_ball_positions = ballDetector.getBallColorPositions(warped, raw_ball_positions)

            drawColoredBallPositions(colored_ball_positions, warped)

            cv.imshow("frame", frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                Settings.RECORDING_TABLE = False
            elif key == ord('p') or Settings.TAKE_A_PICTURE_EVENT.is_set():
                Utils.PictureManager.takePictureAndSave(warped)

    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting...")

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        saveCache()
        camera.release()
        cv.destroyAllWindows()




if __name__ == "__main__":
    main()