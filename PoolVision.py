import cv2 as cv
import time, os, traceback
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
            traceback.print_exc()

        finally:
            saveCache()
            self.camera.release()
            cv.destroyAllWindows()

    def processFrames(self):
        try:
            self.convertFrameToWarpedFrame()
            self.getModelPredictions()
            if self.show_simulated_table:
                self.showSimulatedTable()
            self.handleKeyboardInputs()
        except Exception as e:
            print(f"Error in processFrames: {e}")
            traceback.print_exc()

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
        # cv.imshow("Warped Frame", self.warped_frame)

    def getModelPredictions(self):
        results = self.model.predict(self.warped_frame, conf=0.5, verbose=False)
        processed_results = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = self.model.names[cls_id]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])

                processed_results.append({
                    'name': class_name,
                    'box': [x1, y1, x2, y2],
                    'confidence': conf
                })

                # Optional: draw box overly
                self.drawBoxOverlay(x1, x2, y1, y2)

        self.model_results = processed_results

    def drawBoxOverlay(self, x1, x2, y1, y2):
        # Draw a faint rectangle border (no fill)
        box_color = (180, 180, 180)  # Light gray (BGR)
        thickness = 2  # Thin border
        cv.rectangle(self.warped_frame, (x1, y1), (x2, y2), box_color, thickness)

    def handleKeyboardInputs(self):
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            Settings.ALLOW_TABLE_RECORDING = False
        elif key == ord('p') or Settings.TAKE_A_PICTURE_EVENT.is_set():
            Utils.PictureManager.takePictureAndSave(self.warped_frame)

