import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Class to store detected points and the active cameras which have detected them
class Scene:
    def __init__(self, cameras, detector):
        self.points = []
        self.active_cameras = []
        self.points = self.detect_markers(cameras, detector)

    # Method to detect markers given a list of cameras and detectors
    def detect_markers(self, cameras, detector):
        points = []
        self.active_cameras = []
        for camera in cameras:
            corners, _, _ = camera.detect_aruco_markers(detector)
            corners = np.array(corners)
            if len(corners) == 4:  # No of markers detected should be 4
                corners = np.concatenate([arr.reshape(-1, 2) for arr in corners])
                points.append(corners)
                self.active_cameras.append(camera)

        return points

    # Method to update 2D points, exponentially averaged with new points
    def update_points(self, cameras, detector):
        points = self.detect_markers(cameras, detector)
        alpha = 0.1
        self.points = alpha*points + (1-alpha)*self.points


    # Method to return mean position of all points
    def mean_position(self):
        points_3d = self.map_points()
        meanPos = np.zeros(3)
        for point in points_3d:
            meanPos += point
        meanPos /= len(points_3d)
        return meanPos
