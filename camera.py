import cv2
from config import CANVAS_WIDTH, CANVAS_HEIGHT

class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(3, CANVAS_WIDTH)
        self.cap.set(4, CANVAS_HEIGHT)

    def read(self):
        success, frame = self.cap.read()
        if not success:
            return None
        # Return raw frame, main will handle flipping/processing
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
