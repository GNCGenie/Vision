import cv2 as cv
import numpy as np

# Camera object with methods to grab images and detect aruco markers
class Camera:
    # Initialise camera with camera matrix and distortion coefficients
    def __init__(self, K, d, video_source):
        self.address = video_source
        self.K = K
        self.d = d
        self.R = np.zeros(3)
        self.t = np.zeros(3)

        self.video_capture = cv.VideoCapture(self.address)
        assert self.video_capture.isOpened(), f"Error opening video source: {video_source}"

    # Method to reload camera
    def reload(self):
        self.video_capture.release()
        self.video_capture = cv.VideoCapture(self.address)

    # Method to set rotation and translation vectors
    def set_Rt(self, R, t):
        self.R = R
        self.t = t

    # Grab image from camera
    def grab_image(self):
        valid, image = self.video_capture.read()
        if not valid:
            raise RuntimeError("Failed to capture image from camera")
        return image

    # Detect aruco markers and return corners, ids and rejected
    def detect_aruco_markers(self, detector):
        image = self.grab_image()
        corners, ids, rejected = detector.detectMarkers(image)
        return corners, ids, rejected

    # Project points using camera parameters
    def project(self, points_3d):
        image_points, _ = cv.projectPoints(points_3d, self.R, self.t, self.K, self.d)
        return image_points

# Camera matrix and distortion coefficients
K = np.float32([[1.23637547e+03, 0.00000000e+00, 2.94357998e+02], # IMX335
                [0.00000000e+00, 1.22549758e+03, 2.20521015e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
d = np.float32([-1.5209742,   2.46986782,  0.07634431,  0.07220847,  1.9630171 ])

def load_cameras():
    cameras = []
    for i in range(4):
        print(f"Loading camera {i}")
        try:
            cameras.append(Camera(K,d, f"/dev/video{i}"))
        except AssertionError as e:
            print(e)
    print(f"Using {len(cameras)} cameras")
    return cameras
