import cv2 as cv
import os, pygame, time, vosk
import sounddevice as sd
import queue, json, threading
import config.settings as Settings

class VoiceManager:
    @staticmethod
    def start_listening(model_path, device_index, print_audio):
        """
        Start the voice listener in a background thread.
        
        Args:
            model_path (str): Path to the vosk model directory.
            device_index (int): Microphone device index.
            trigger_callback (function): Function to call when trigger phrase detected.
        """
        def listen():
            try:
                print(f"Loading Vosk model from {model_path}")
                model = vosk.Model(model_path)
                recognizer = vosk.KaldiRecognizer(model, 16000)
                audio_queue = queue.Queue()

                def audio_callback(indata, frames, time, status):
                    if status:
                        print(status)
                    audio_queue.put(bytes(indata))

                print("Opening audio stream...")
                with sd.RawInputStream(device=device_index, samplerate=16000, blocksize=8000, dtype='int16',
                                       channels=1, callback=audio_callback):
                    print("Listening (Vosk)...")
                    while True:
                        data = audio_queue.get()
                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            command = result.get("text", "").lower()
                            if command:
                                if print_audio: print(f"You said: {command}")
                                if "take a picture" in command:
                                    Settings.TAKE_A_PICTURE_EVENT.set()
                                

            except Exception as e:
                print(f"Vosk thread error: {e}")
                import traceback
                traceback.print_exc()

        # Start as background thread
        voice_thread = threading.Thread(target=listen, daemon=True)
        voice_thread.start()

    @staticmethod
    def toggleAudioCommands(print_audio = False):
        if Settings.LISTEN_FOR_AUDIO_COMMANDS:
            VoiceManager.start_listening(
                model_path=r"C:\Users\abdul\Downloads\vosk-model-small-en-us-0.15",
                device_index=5,
                print_audio = print_audio
            )

class PictureManager:
    @staticmethod
    def takePictureAndSave(frame):
        try:
            machine_learning_path = "Images/machine_learning_images"
            os.makedirs(machine_learning_path, exist_ok=True)  # Ensure the directory exists
            filename = f"machine_learning_{int(time.time() * 1000)}.jpg"
            filepath = os.path.join(machine_learning_path, filename)
            cv.imwrite(filepath, frame)
            PictureManager.playCameraShutterSound()
            Settings.TAKE_A_PICTURE_EVENT.clear() # Reset the event
        except Exception as e:
            print(f"Failed to save picture: {e}")

    @staticmethod
    def playCameraShutterSound():
        try:
            pygame.mixer.init()
            sound_path = "SoundEffects/shutter.wav"
            pygame.mixer.Sound(sound_path).play()
        except Exception as e:
            print(f"Shutter sound failed: {e}")
