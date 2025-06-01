import cv2 as cv
import time, os, pygame
import config.settings as Settings
import utils.Utils as Utils
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame, saveCache
from processing.BallDetector import initializeBallDetector, drawColoredBallPositions, drawColoredBallPositionsOnNewFrame
from ultralytics import YOLO
import pygame

def main():
    print("START OF FILE")
    Utils.VoiceManager.toggleAudioCommands(print_audio=True)
    camera = CameraCapture(camera_index=0, width=Settings.WARPED_TABLE_W, height=Settings.WARPED_TABLE_H)
    ballDetector = initializeBallDetector()
    model = YOLO('model/runs/detect/train/weights/best.pt')
    print("GRabbed model")
    try:
        while Settings.RECORDING_TABLE:
            print("MAIN LOOP RUNNING")
            frame = camera.get_frame()
            if frame is None:
                print("Failed to capture frame.")
                continue

            warped = getWarpedFrame(frame, debug_mode=False)
            if warped is None:
                print("Failed to warp frame.")
                continue

            results = model.predict(warped, conf=0.5)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Draw on warped frame
                    cv.rectangle(warped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv.putText(warped, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            colored_ball_positions = ballDetector.getBallColorPositionsFromModel(results, Settings.COLOR_MAP)
            ball_frame = drawColoredBallPositionsOnNewFrame(colored_ball_positions, warped, Settings.COLOR_MAP)
            cv.imshow("Ball Positions", ball_frame)
            cv.imshow("Warped Frame", warped)

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