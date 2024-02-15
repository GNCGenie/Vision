import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import cv2
from copy import deepcopy
import time
from n_cam_extrinsic_calculation import getExtrinsics

############################################################
############################################################
############################################################
############################################################
# Visualize the reconstructed 3D points
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Axis for plotting 3D points
scatter = ax.scatter([], [], [], c='r', marker='o')
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_zlim(-0.0, 2.0)
ax.grid(True)
# Axis for 2D plotting camera views
ax2d_cam1 = fig.add_subplot(331)
ax2d_cam2 = fig.add_subplot(333)
ax2d_cam3 = fig.add_subplot(337)
ax2d_cam4 = fig.add_subplot(339)
for ax2d in [ax2d_cam1, ax2d_cam2, ax2d_cam3, ax2d_cam4]:
    ax2d.set_aspect('equal')
    ax2d.set_xlabel('X [pixels]')
    ax2d.set_ylabel('Y [pixels]')
    ax2d.grid(True)
scatter2d_cam1 = ax2d_cam1.scatter([], [], c='r', marker='o')
scatter2d_cam2 = ax2d_cam2.scatter([], [], c='r', marker='o')
scatter2d_cam3 = ax2d_cam3.scatter([], [], c='r', marker='o')
scatter2d_cam4 = ax2d_cam4.scatter([], [], c='r', marker='o')
scatter2d = [scatter2d_cam1, scatter2d_cam2, scatter2d_cam3, scatter2d_cam4]
plt.show()  # Show the initial plot
############################################################
############################################################
############################################################
############################################################

def getPoints():
    ############################################################
    # Feature Detection
    active_cameras = 0

    disp=[]
    for i in range(n_cameras):
        _, image = video_captures[i]()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Chessboard pattern detection
#       board_pattern = (9,6)
#       complete, pts_i = cv2.findChessboardCorners(image, board_pattern)

        # Aruco marker detection
        pts_i, _, _ = detector.detectMarkers(image)
        image = cv2.aruco.drawDetectedMarkers(image, pts_i)

        image = cv2.resize(image, (512, 384))
        disp.append(image)

        if (pts_i is not None and len(pts_i) == n_markers):
            pts_i = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
            active_cameras += 1
            pts[:, :, i] = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
        else:
            continue

    # Join 4 images in disp in a 2x2 grid
    disp_grid = np.vstack((np.hstack((disp[0], disp[1])), np.hstack((disp[2], disp[3]))))
    cv2.imshow('Display', disp_grid)
    cv2.waitKey(1)
    return pts, active_cameras

def project(points_3d, rvec, tvec, K):
    ############################################################
    # Projection 3D points to 2D for each camera
    proj_points = Rotation.from_rotvec(rvec).apply(points_3d)
    proj_points += tvec
    proj_points = proj_points @ K.T
    proj_points /= proj_points[2, np.newaxis]
    return proj_points[:, :2]

def cost_func(var, rvecs, tvecs, pts, K, d, n_points, n_cameras):
    ############################################################
    # Cost function reprojection
    X = var[:n_points*3].reshape(n_points, 3)

    err = np.zeros((n_points*2, n_cameras))
    for i in range(n_cameras):
        proj = cv2.projectPoints(X, rvecs[i], tvecs[i], K, d)[0].reshape(-1,2)
#       proj = project(X, rvecs[i], tvecs[i], K)
        err[:,i] = (proj-pts[:,:,i]).ravel()

    return err.ravel()

# Aruco dictionary being used for each camera
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
# Camera matrix and distortion coefficients
K = np.float32([[2e3, 0, 1296],
                [0, 2e3, 972],
                [0, 0, 1]])
d = np.float32([-0.450,
                0.2553,
                -0.000,
                -0.000,
                -0.085])
# Create lists to hold camera objects and video captures
n_markers = 4
n_points = 4*n_markers
n_cameras = 4  # Change this to the desired number of cameras
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

alpha = 0.1
pts = np.zeros((n_points, 2, n_cameras))
pts_prev = np.zeros((n_points, 2, n_cameras))
X = np.ones((n_points, 3))
rvecs,tvecs = getExtrinsics(video_captures, 54, 54, n_cameras)
#rvecs,tvecs = np.random.rand(n_cameras, 3), np.random.rand(n_cameras, 3)
cost = np.inf

while True:
    ############################################################
    # Get and update points
    start_time = time.time()
    pts, active_cameras = getPoints()
    if active_cameras < 2:
        print(f'Not enough active cameras {active_cameras}', end='\r')
        continue
    pts = alpha * pts + (1 - alpha) * pts_prev
    pts_prev = pts
    print("Time to get points: %s" % (time.time() - start_time))

    ############################################################
    # Triangulate points first for rough estimate
    start_time = time.time()
    R1 = cv2.Rodrigues(rvecs[0])[0]
    P1 = K @ np.vstack((R1, tvecs[0])).T
    R2 = cv2.Rodrigues(rvecs[1])[0]
    P2 = K @ np.vstack((R2, tvecs[1])).T
    X = cv2.triangulatePoints(P1, P2, pts[:,:,0].T, pts[:,:,1].T)
    X /= X[3]
    X = X[:3].T
    print("Time to triangulate: %s" % (time.time() - start_time))

    ############################################################
    # Optimization
    start_time = time.time()
    var = np.concatenate([X.ravel()])
#    if cost < 3e3:
#    else:
#        var = np.random.rand(n_points * 3)
    solution = least_squares(cost_func, var, args=(rvecs, tvecs, pts, K, d, n_points, n_cameras),
#                             method='trf',
#                             loss='cauchy',
#                             bounds=(-2, 2),
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             max_nfev=1000)
    cost = np.linalg.norm(solution.fun)
    print('Cost = {}'.format(cost))
    print("Time to optimize: %s" % (time.time() - start_time))
    optimized_vars = solution.x
    if np.isnan(optimized_vars).any() or np.isinf(optimized_vars).any():
        continue # If any values in res are NaN or Inf, continue
    var = optimized_vars

    ############################################################
    # Recollect X, rvecs, tvecs from res
    start_time = time.time()
    X = var[:n_points * 3].reshape(n_points, 3)
    print('Mean position of 3D points = {}'.format(np.mean(X, axis=0)))
    print("Time to recollect: %s" % (time.time() - start_time))

############################################################
############################################################
############################################################
############################################################
    # Add 2D points from cam1 and cam2 to the plot
    for i,ax in zip(range(n_cameras), [ax2d_cam1, ax2d_cam2, ax2d_cam3, ax2d_cam4]):
        ax.set_xlim(min(pts[:, 0, i]), max(pts[:, 0, i]))
        ax.set_ylim(min(pts[:, 1, i]), max(pts[:, 1, i]))
        # Set new data
        scatter2d[i].set_offsets(pts[:, :, i])

    # Add 3D points triangulated to the plot
    scatter._offsets3d = (X[:,0], X[:,1], X[:,2])
    plt.draw()  # Redraw the plot
    plt.pause(0.001)
############################################################
############################################################
############################################################
############################################################
