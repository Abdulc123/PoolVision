import cv2

class CameraCapture:
    def __init__(self, camera_index = 1, width = 1280, height = 720):
        """
        Initializes the webcam
        :param cameraIndex: 1 for the secondary external camera
        :param width: Width of the video frame
        :parm height: Height of the video frame
        """
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
        Frees up the webcame connection cv2 made when done with window
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

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1)
            # Closes window when q is pressed or close is selected
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.release()
        cv2.destroyAllWindows()