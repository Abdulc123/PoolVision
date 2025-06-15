import cv2 as cv
import time, os, pygame
import config.settings as Settings
import utils.Utils as Utils
from display.simulated_table import SimulatedTable
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame, saveCache
from ultralytics import YOLO
import pygame

class PoolVision:
    def __init__(self, show_simulated_table):
        self.show_simulated_table = show_simulated_table

        Utils.VoiceManager.toggleAudioCommands(print_audio=True)
        self.simulatedTable = SimulatedTable()
        self.camera = CameraCapture(camera_index=0, width=Settings.WARPED_TABLE_W, height=Settings.WARPED_TABLE_H)
        self.model = YOLO('model/runs/detect/train/weights/best.pt')

        self.warped_frame = None
        self.model_results = None

    def start(self):
        try:
            while Settings.ALLOW_TABLE_RECORDING:
                self.processFrames()

        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")

        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            saveCache()
            self.camera.release()
            cv.destroyAllWindows()

    def processFrames(self):
        self.convertFrameToWarpedFrame()
        self.getModelPredictions()

        if self.show_simulated_table:
            self.showSimulatedTable()

        self.handleKeyboardInputs()

    def convertFrameToWarpedFrame(self):
        frame = self.camera.getFrame()
        if frame is None:
            print("Failed to capture frame.")
            return

        warped_frame = getWarpedFrame(frame, debug_mode=False)
        if warped_frame is None:
            print("Failed to warp frame.")
            return
        
        self.warped_frame = warped_frame
    
    def showSimulatedTable(self):
        simulated_frame = self.simulatedTable.getSimulatedTableFrame(self.model_results, self.warped_frame)
        cv.imshow("Simulated Table", simulated_frame)
        cv.imshow("Warped Frame", self.warped_frame)

    def getModelPredictions(self):
        results = self.model.predict(self.warped_frame, conf=0.5, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Draw on warped frame
                cv.rectangle(self.warped_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv.putText(self.warped_frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.model_results = results

    def handleKeyboardInputs(self):
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            Settings.ALLOW_TABLE_RECORDING = False
        elif key == ord('p') or Settings.TAKE_A_PICTURE_EVENT.is_set():
            Utils.PictureManager.takePictureAndSave(self.warped_frame)

