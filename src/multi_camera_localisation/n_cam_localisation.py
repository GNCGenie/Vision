import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import cv2
import time
import itertools
from n_cam_extrinsic_calculation import get_extrinsics
from plotting import *

import concurrent.futures
def get_points_parallel(pts, detector):
    active_cameras = []
    n_cameras = len(pts[0, 0, :])
    n_points = len(pts[:, 0, 0])
    def process_image(i):
        _, image = video_captures[i]()
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Aruco marker detection
        pts_i, _, _ = detector.detectMarkers(image)
        if (pts_i is not None and len(pts_i) == n_points):
            active_cameras.append(i)
            pts[:, :, i] = np.concatenate([np.mean(arr, axis=1) for arr in pts_i])
            return pts_i
        else:
            return None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, i) for i in range(n_cameras)]
        for i, future in enumerate(futures):
            result = future.result()
    active_cameras.sort()
    return pts, active_cameras

def cost_func(var, rvecs, tvecs, pts, K, d, n_points, n_cameras):
    ############################################################
    # Cost function reprojection
    X = var[:n_points*3].reshape(n_points, 3)

    err = np.zeros((n_points*2, n_cameras))
    for i in range(n_cameras):
        proj = cv2.projectPoints(X, rvecs[i], tvecs[i], K, d)[0].reshape(-1,2)
        err[:,i] = (proj-pts[:,:,i]).ravel()

    return err.ravel()

# Aruco dictionary being used for each camera
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
# Camera matrix and distortion coefficients
resX, resY = 2592, 1944
fx = fy = 2e3
K = np.float32([[fx, 0, resX/2],
                [0, fy, resY/2],
                [0, 0, 1]])
d = np.float32([-0.45,
                0.255,
                -0.00,
                -0.00,
                -0.08])
# Initialize cameras and video captures
n_cameras = 3  # Change this to the desired number of cameras
video_captures = []
for i in range(0, 20):
    cam = cv2.VideoCapture(i)  # Adjust camera indices as needed
    # Capture cam reading error
    if not cam.isOpened():
        continue
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cam.set(cv2.CAP_PROP_EXPOSURE , 1e0) # Change if too dark or too bright
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resX)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resY)
    video_captures.append(cam.read)  # Capture initial frames
    if len(video_captures) == n_cameras:
        print(f'Connected to {video_captures} cameras')
        break

# Create lists to hold camera objects and video captures
n_points = 1 # Change this to set the no. of markers/LEDs that are going to be detected in the scene
alpha = 0.10
pts = np.zeros((n_points, 2, n_cameras))
pts_prev = np.zeros((n_points, 2, n_cameras))
X = np.ones((n_points, 3))
rvecs,tvecs = get_extrinsics(video_captures, 54, 54, n_cameras)
cost = np.inf

while True:
    ############################################################
    # Get and update points
    start_time = time.time()
    pts, active_cameras = get_points_parallel(pts, detector)
    if len(active_cameras) < 2:
        print(f'Not enough active cameras {active_cameras}', end='\r')
        continue
    pts = alpha * pts + (1 - alpha) * pts_prev
    pts_prev = pts
#    print("Time to get points: %s" % (time.time() - start_time))

    ############################################################
    # Triangulate points first for rough estimate
    if cost > 1e2:
#        start_time = time.time()
        X = np.zeros((n_points, 3))
        for i,j in itertools.combinations(active_cameras, 2):
            ptsi = cv2.undistortPoints(pts[:,:,i], K, d).reshape(-1,2)
            ptsj = cv2.undistortPoints(pts[:,:,j], K, d).reshape(-1,2)
            Pi = np.vstack((cv2.Rodrigues(rvecs[i])[0], tvecs[i])).T
            Pj = np.vstack((cv2.Rodrigues(rvecs[j])[0], tvecs[j])).T
            Xtmp = cv2.triangulatePoints(Pi, Pj, ptsi.T, ptsj.T)
            Xtmp /= Xtmp[3]
            Xtmp = Xtmp[:3].T
            X += Xtmp
#        print("Time to triangulate: %s" % (time.time() - start_time))
        X = X / len(active_cameras)

    ############################################################
    # Optimization
#    start_time = time.time()
    var = np.concatenate([X.ravel()])
    solution = least_squares(cost_func, var, args=(rvecs[active_cameras], tvecs[active_cameras],
                                                   pts[:,:,active_cameras], K, d,
                                                   n_points, len(active_cameras)),
#                             method='trf',
#                             bounds=(-1e1, 1e1),
#                             loss='cauchy',
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             max_nfev=1000)
    cost = np.linalg.norm(solution.fun)
    optimized_vars = solution.x
    if np.isnan(optimized_vars).any() or np.isinf(optimized_vars).any():
        continue # If any values in res are NaN or Inf, continue
    var = optimized_vars
#    print("Time to optimize: %s" % (time.time() - start_time))
    print('Cost = {}'.format(cost), end='\t')

    ############################################################
    # Recollect X from res
    X = var[:n_points * 3].reshape(n_points, 3)
    print("Time to complete: %s" % (time.time() - start_time), end='\t')
    print('Position of 3D points = {}'.format(X), end='\n')

    ############################################################
#    visualize(pts, X)
