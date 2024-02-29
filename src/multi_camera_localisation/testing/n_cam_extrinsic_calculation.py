import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import cv2
from copy import deepcopy
import time

def get_extrinsics(video_captures, n_markers, n_points, n_cameras):
    n_cameras = len(video_captures)
    import concurrent.futures
    def get_points_parallel(pts):
        active_cameras = []
        def process_image(i):
            _, image = video_captures[i]()
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Chessboard pattern detection
            board_pattern = (9,6)
            complete, pts_i = cv2.findChessboardCorners(image, board_pattern)
            if (pts_i is not None and len(pts_i) == n_markers):
                pts_i = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
                active_cameras.append(i)
                pts[:, :, i] = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
                return pts_i
            else:
                return None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, i) for i in range(n_cameras)]
            for i, future in enumerate(futures):
                result = future.result()
        return pts, active_cameras

    def cost_func(var, X, pts, K, d, n_points, n_cameras):
        ############################################################
        # Cost function reprojection
        rvecs = var[:n_cameras * 3].reshape(-1, 3)
        tvecs = var[n_cameras * 3:].reshape(-1,3)

        err = np.zeros((n_points*2, n_cameras))
        for i in range(n_cameras):
            proj = cv2.projectPoints(X, rvecs[i], tvecs[i], K, d)[0].reshape(-1,2)
            err[:,i] = (proj-pts[:,:,i]).ravel()

        return err.ravel()

    K = np.float32([[2e3, 0, 1296],
                    [0, 2e3, 972],
                    [0, 0, 1]])
    d = np.float32([-0.450,
                    0.2553,
                    -0.000,
                    -0.000,
                    -0.085])

    board_pattern = (9, 6)
    board_cellsize = 0.02475
    X = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    X = np.array(X, dtype=np.float32) * board_cellsize

    pts = np.zeros((n_points, 2, n_cameras))
    pts, active_cameras = get_points_parallel(pts)
    while len(active_cameras) != n_cameras:
        print(f'Not enough active cameras {active_cameras}', end='\r')
        pts, active_cameras = get_points_parallel(pts)
    rvecs, tvecs = np.ones((n_cameras, 3)), np.ones((n_cameras, 3))
    # Initialise rvecs and tvecs by running solvePnP
    for i in range(n_cameras):
        ret, rvec, tvec = cv2.solvePnP(X, pts[:,:,i], K, d)
        rvec = rvec.reshape(3)
        tvec = tvec.reshape(3)
        rvecs[i], tvecs[i] = rvec, tvec

    alpha = 0.05
    pts_prev = deepcopy(pts)
    t0 = time.time()
    while time.time() - t0 < 2:
        ############################################################
        # Get and update points
        start_time = time.time()
        pts, active_cameras = get_points_parallel(pts)
        if len(active_cameras) < 2:
            print(f'Not enough active cameras {active_cameras}', end='\r')
            continue
        pts = alpha * pts + (1 - alpha) * pts_prev
        pts_prev = pts
        print("Time to get points: %s" % (time.time() - start_time))

        ############################################################
        # Optimization
        start_time = time.time()
        var = np.concatenate([rvecs.ravel(), tvecs.ravel()])
        solution = least_squares(cost_func, var, args=(X, pts, K, d, n_points, n_cameras),
#                                method='trf',
#                                loss='cauchy',
#                                bounds=(-2, 2),
                                ftol=1e-15, xtol=1e-15, gtol=1e-15,
                                max_nfev=1000)
        print('Cost = {}'.format(np.linalg.norm(solution.fun)))
        print("Time to optimize: %s" % (time.time() - start_time))
        optimized_vars = solution.x
        if np.isnan(optimized_vars).any() or np.isinf(optimized_vars).any():
            continue # If any values in res are NaN or Inf, continue
        var = optimized_vars

        ############################################################
        # Recollect X, rvecs, tvecs from res
        start_time = time.time()
        rvecs = var[:n_cameras * 3].reshape(-1,3)
        tvecs = var[n_cameras * 3:].reshape(-1,3)
        # Print the 3D position, rotation vectors and translation vectors:
        for i in range(n_cameras):
            print('Rotation vector of camera {} = {}'.format(i, rvecs[i]))
            print('Translation vector of camera {} = {}'.format(i, tvecs[i]))
        print("Time to recollect: %s" % (time.time() - start_time))

    return rvecs, tvecs

if __name__ == '__main__':
    n_cameras = 3
    # Initialize cameras and video captures
    video_captures = []
    for i in range(0, 10):
        cam = cv2.VideoCapture(i)  # Adjust camera indices as needed
        # Capture cam reading error
        if not cam.isOpened():
            continue
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cam.set(cv2.CAP_PROP_EXPOSURE , 1e0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        video_captures.append(cam.read)  # Capture initial frames
        if len(video_captures) == n_cameras:
            print(f'Connected to {video_captures} cameras')
            break

    rvecs,tvecs = get_extrinsics(video_captures, 54, 54, n_cameras)
    print('rvecs = {}'.format(rvecs))
    print('tvecs = {}'.format(tvecs))
