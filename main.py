import cv2 as cv
import speech_recognition as sr
import simpleaudio as sa
import time, os, threading, pygame
import config.settings as Settings
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame, saveCache
from processing.BallDetector import BallDetector
from display.screenDisplay import drawBallPositions, drawColoredBallPositions
import pygame


# def listenForVoice():
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()

#     with mic as source:
#         recognizer.adjust_for_ambient_noise(source)
    
#     while True:
#         with mic as source:
#             print("Listening...")
#             audio = recognizer.listen(source)
#         try:
#             command = recognizer.rec(audio).lower()
#             print(f"You said: {command}")
#             if "take a picture" in command:
#                 Settings.TAKE_A_PICTURE_EVENT.set() # Signal the event
#         except sr.UnknownValueError:
#             pass
#         except sr.RequestError as e:
#             print(f"Audio API error: {e}")


# # Start voice listener in a background thread
# if Settings.LISTEN_FOR_AUDIO_COMMANDS:
#     voice_thread = threading.Thread(target=listenForVoice, daemon=True)
#     voice_thread.start()


def main():
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
                takePictureAndSave(warped)

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


def takePictureAndSave(frame):
    try:
        machine_learning_path = "Images/machine_learning_images"
        os.makedirs(machine_learning_path, exist_ok=True)  # Ensure the directory exists
        filename = f"machine_learning_{int(time.time() * 1000)}.jpg"
        filepath = os.path.join(machine_learning_path, filename)
        cv.imwrite(filepath, frame)
        playCameraShutterSound()
        Settings.TAKE_A_PICTURE_EVENT.clear() # Reset the event
    except Exception as e:
        print(f"Failed to save picture: {e}")

def playCameraShutterSound():
    try:
        pygame.mixer.init()
        sound_path = "SoundEffects/shutter.wav"
        pygame.mixer.Sound(sound_path).play()
    except Exception as e:
        print(f"Shutter sound failed: {e}")



if __name__ == "__main__":
    main()