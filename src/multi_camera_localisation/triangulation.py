import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from camera_handling import load_cameras
from scene import DetectedPoints

def set_rel_pose(cameras, points, active_cameras):
    for camera in cameras[1:]:
        print(f"Solving for C{camera.id} with C{0}")
        F, _ = cv2.findFundamentalMat(points[0], points[camera.id], cv2.FM_8POINT)
        E = camera.K.T @ F @ camera.K

        _, R, t, _ = cv2.recoverPose(E, points[0], points[camera.id])
        rvec = Rotation.from_matrix(R).as_rotvec()
        camera.set_Rt(rvec,t)

#    for j in active_cameras[1:]:
#        print(f"Solving for C{0} with C{j}")
#        F, _ = cv2.findFundamentalMat(points[0], points[j], cv2.FM_8POINT)
#        E = cameras[j].K.T @ F @ cameras[j].K
#
#        _, R, t, _ = cv2.recoverPose(E, points[0], points[j])
#        rvec = Rotation.from_matrix(R).as_rotvec()
#        cameras[j].set_Rt(rvec,t)
#        print("Rotation", rvec)
#        print("Translation", t)

# Method to initialise cameras
def initialise_cameras(cameras,detector):
    scene = DetectedPoints(cameras, detector)
    if len(scene.active_cameras) < 2:
        raise RuntimeError("Not enough cameras active")

    # Estimate relative pose between cameras
    set_rel_pose(cameras, scene.points, scene.active_cameras)
