import cv2

class CameraCapture:
    def __init__(self, camera_index = 1, width = 1280, height = 720):
        """
        Initializes the webcam
        :param cameraIndex: 1 for the secondary external camera
        :param width: Width of the video frame
        :parm height: Height of the video frame
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError("CameraCapture: Failed to open camera.")
        
    def get_frame(self):
        """
        Capture a single frame from the camera.
        :return: The captured frame or None if it failed
        """