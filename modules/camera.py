import cv2
import threading
import time

class StereoCamera:
    def __init__(self, left_id=0, right_id=1, width=640, height=480, fps=30):
        self.left_id = left_id
        self.right_id = right_id
        self.width = width
        self.height = height
        self.fps = fps
        self.stopped = False
        
        # Initialize cameras
        self.cap_left = cv2.VideoCapture(self.left_id)
        self.cap_right = cv2.VideoCapture(self.right_id)
        
        self._configure_camera(self.cap_left)
        self._configure_camera(self.cap_right)
        
        self.left_frame = None
        self.right_frame = None
        self.ret_left = False
        self.ret_right = False
        
        # Lock for thread safety
        self.lock = threading.Lock()

    def _configure_camera(self, cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Low buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        """Start the thread to read frames from the video stream."""
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        # Warmup
        time.sleep(1.0)
        return self

    def update(self):
        """Keep looping infinitely until the thread is stopped."""
        while not self.stopped:
            if not self.cap_left.isOpened() or not self.cap_right.isOpened():
                time.sleep(0.1)
                continue
                
            ret_l, frame_l = self.cap_left.read()
            ret_r, frame_r = self.cap_right.read()
            
            with self.lock:
                self.ret_left = ret_l
                self.left_frame = frame_l
                self.ret_right = ret_r
                self.right_frame = frame_r

    def read(self):
        """Return the most recent frame tuple."""
        with self.lock:
            return (self.ret_left, self.left_frame, self.ret_right, self.right_frame)

    def release(self):
        self.stopped = True
        time.sleep(0.2)
        self.cap_left.release()
        self.cap_right.release()

if __name__ == "__main__":
    # Test
    cam = StereoCamera(left_id=0, right_id=1)
    cam.start()
    time.sleep(2)
    ret_l, l, ret_r, r = cam.read()
    if ret_l:
        print(f"Left frame shape: {l.shape}")
    if ret_r:
        print(f"Right frame shape: {r.shape}")
    cam.release()
