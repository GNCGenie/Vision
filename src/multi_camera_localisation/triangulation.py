import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from camera_handling import load_cameras
from scene import Scene

# Method to recover camera pose of all active camera and update r and t
def set_rel_pose(cameras, points, active_cameras):

    for i,camera in enumerate(active_cameras):
        print(f"Solving for C{i} with C{0}")

        F, _ = cv2.findFundamentalMat(points[0], points[i], cv2.FM_8POINT)
        E = camera.K.T @ F @ camera.K

        _, R, t, _ = cv2.recoverPose(E, points[0], points[i])
        rvec = Rotation.from_matrix(R).as_rotvec()
        camera.set_Rt(rvec,t)

        _, E, R, t, _ = cv2.recoverPose(points[0], points[i],
                                        active_cameras[0].K, active_cameras[0].d,
                                        camera.K, camera.d)

# Method to initialise cameras
def initialise_cameras(cameras,detector):
    if len(cameras) < 2:
        raise RuntimeError("Not enough cameras active")

    scene = Scene(cameras, detector)
    if len(scene.active_cameras) < 2:
        print("WARNING: ", len(scene.active_cameras), "cameras detecting the points (not enough)")
        return

    # Estimate relative pose between cameras
    set_rel_pose(cameras, scene.points, scene.active_cameras)
