import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import cv2
from copy import deepcopy
import time

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
for ax2d in [ax2d_cam1, ax2d_cam2, ax2d_cam3]:
    ax2d.set_aspect('equal')
    ax2d.set_xlabel('X [pixels]')
    ax2d.set_ylabel('Y [pixels]')
    ax2d.grid(True)
scatter2d_cam1 = ax2d_cam1.scatter([], [], c='r', marker='o')
scatter2d_cam2 = ax2d_cam2.scatter([], [], c='r', marker='o')
scatter2d_cam3 = ax2d_cam3.scatter([], [], c='r', marker='o')
scatter2d = [scatter2d_cam1, scatter2d_cam2, scatter2d_cam3]
plt.show()  # Show the initial plot
############################################################
############################################################
############################################################
############################################################

# Aruco dictionary being used for each camera
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
# Camera matrix and distortion coefficients
K = np.float32([[1997.261510455484, 0, 1228.67838185292], # IMX335 4K
                [0, 1983.185627905062, 963.4249587878591],
                [0, 0, 1]])
K = np.float32([[2e3, 0, 1296],
                [0, 2e3, 972],
                [0, 0, 1]])
d = np.float32([-0.4500085544835851, 0.2553452984680251, -0.0007677480699666998, -0.0005839423190390148, -0.08536976101180629])
# Create lists to hold camera objects and video captures
n_markers = 54
n_points = 1*n_markers
n_cameras = 3  # Change this to the desired number of cameras
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
        break

def getPoints():
    ############################################################
    # Feature Detection
    active_cameras = 0

    disp=[]
    for i in range(n_cameras):
        _, image = video_captures[i]()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Chessboard pattern detection
        board_pattern = (9,6)
        complete, pts_i = cv2.findChessboardCorners(image, board_pattern)

        # Aruco marker detection
#       pts_i, _, _ = detector.detectMarkers(image)
#       image = cv2.aruco.drawDetectedMarkers(image, pts_i)

        image = cv2.resize(image, (512, 384))
        disp.append(image)

        if (pts_i is not None and len(pts_i) == n_markers):
            pts_i = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
            active_cameras += 1
            pts[:, :, i] = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
        else:
            continue

    # Join all images in display
    disp = np.concatenate(disp, axis=1)
    cv2.imshow('Display', disp)
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

def cost_func(var, X, pts, K, d, n_points, n_cameras):
    ############################################################
    # Cost function reprojection
    rvecs = var[:n_cameras * 3].reshape(-1, 3)
    tvecs = var[n_cameras * 3:].reshape(-1,3)

    err = np.zeros((n_points*2, n_cameras))
    for i in range(n_cameras):
        proj = cv2.projectPoints(X, rvecs[i], tvecs[i], K, d)[0].reshape(-1,2)
#       proj = project(X, rvecs[i], tvecs[i], K)
        err[:,i] = (proj-pts[:,:,i]).ravel()

    return err.ravel()

alpha = 0.05
pts = np.zeros((n_points, 2, n_cameras))
pts_prev = np.zeros((n_points, 2, n_cameras))
board_pattern = (9, 6)
board_cellsize = 0.02475
X = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
X = np.array(X, dtype=np.float32) * board_cellsize
rvecs = np.random.rand(n_cameras, 3)
tvecs = np.random.rand(n_cameras, 3)
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
    # Optimization
    start_time = time.time()
    var = np.concatenate([rvecs.ravel(), tvecs.ravel()])
    solution = least_squares(cost_func, var, args=(X, pts, K, d, n_points, n_cameras),
                             method='trf',
                             ftol=1e-15,
                             loss='cauchy',
                             bounds=(-2, 2),
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
    rvecs = var[:n_cameras * 3].reshape(-1, 3)
    tvecs = var[n_cameras * 3:].reshape(-1,3)
    # Print the 3D position, rotation vectors and translation vectors:
    for i in range(n_cameras):
        print('Rotation vector of camera {} = {}'.format(i, rvecs[i]))
        print('Translation vector of camera {} = {}'.format(i, tvecs[i]))
    print("Time to recollect: %s" % (time.time() - start_time))

############################################################
############################################################
############################################################
############################################################
    # Add 2D points from cam1 and cam2 to the plot
    for i,ax in zip(range(n_cameras), [ax2d_cam1, ax2d_cam2, ax2d_cam3]):
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
