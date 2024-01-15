import numpy as np
import cv2
from camera_handling import Camera
from camera_handling import load_cameras
from triangulation import initialise_cameras


# Aruco dictionary being used for each camera
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cameras = load_cameras()
while True:
    try:
        initialise_cameras(cameras,detector)
    except RuntimeError as e:
        print(e)
        print("Reloading cameras")
        cameras = load_cameras()
