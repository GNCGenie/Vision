import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Class to store detected points and the active cameras which have detected them
class Scene:
    def __init__(self, cameras, detector):
        self.points = []
        self.active_cameras = []
        self.detect_markers(cameras, detector)

    # Method to detect markers given a list of cameras and detectors
    def detect_markers(self, cameras, detector):
        self.points = []
        self.active_cameras = []
        for camera in cameras:
            corners, _, _ = camera.detect_aruco_markers(detector)
            corners = np.array(corners)
            if len(corners) == 4:  # No of markers detected should be 4
                corners = np.concatenate([arr.reshape(-1, 2) for arr in corners])
                self.points.append(corners)
                self.active_cameras.append(camera)

    # Method to map detected points to 3D points
    def map_points(self):
        if len(self.active_cameras) < 2:
            raise RuntimeError("Not enough cameras detecting the points")

        points_3d = []
        undistorted_2d_points = []
        for points,active_camera in zip(self.points,self.active_cameras):
            undistorted_2d_points.append(
                    cv2.undistortPoints(points, active_camera.K, active_camera.d))

        for j in self.active_cameras[1:]:
            P0 = self.active_cameras[0].K @ np.hstack((self.active_cameras[0].R,
                                                       self.active_cameras[0].t))
            P1 = self.active_cameras[j].K @ np.hstack((self.active_cameras[j].R,
                                                       self.active_cameras[j].t))
            X = cv2.triangulatePoints(P0, P1, self.points[0].T, self.points[j].T)
            X /= X[3]
            X = X[:3]
            points_3d.append(X)

        return points_3d

    # Method to return mean position of all points
    def mean_position(self):
        points_3d = self.map_points()
        meanPos = np.zeros(3)
        for point in points_3d:
            meanPos += point
        meanPos /= len(points_3d)
        return meanPos
