import cv2 as cv

#C:\Users\abdul\AppData\Local\Programs\Python\Python313
#C:\Users\abdul\AppData\Local\Programs\Python\Python313\Scripts

class CameraCapture:
    def __init__(self, camera_index = 1, width = 1280, height = 720):
        """
        Initializes the webcam
        :param cameraIndex: 1 for the secondary external camera
        :param width: Width of the video frame
        :parm height: Height of the video frame
        """
        self.cap = cv.VideoCapture(camera_index, cv.CAP_DSHOW)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv.CAP_PROP_FPS, 30)

        # Print out the actual FPS
        actual_fps = self.cap.get(cv.CAP_PROP_FPS)
        print(f"[INFO] Camera running at {actual_fps} FPS")

        if not self.cap.isOpened():
            raise RuntimeError("CameraCapture: Failed to open camera.")
        
    def get_frame(self):
        """
        Capture a single frame from the camera each time its called
        :return: The captured frame or None if it failed
        """
        result, frame = self.cap.read()
        if not result:
            return None
        return frame
    
    def release(self):
        """
        Frees up the webcame connection cv made when done with window
        """
        self.cap.release()

    def show_live_feed(self, window_name = "Pool Vision Live Camera Feed"):
        """
        (Optional) Show a lime camera feed window for debugging.
        Press q to quit
        """
        while True:
            frame = self.get_frame()
            if frame is None:
                print("CameraCapture: Failed to capture frame.")
                break

            cv.imshow(window_name, frame)

            key = cv.waitKey(1)
            # Closes window when q is pressed or close is selected
            if key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                break

        self.release()
        cv.destroyAllWindows()