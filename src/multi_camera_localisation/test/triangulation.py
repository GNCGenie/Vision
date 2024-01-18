import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2
from copy import deepcopy
import time

# Visualize the reconstructed 3D points
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax2d_cam1 = fig.add_subplot(331)
ax2d_cam2 = fig.add_subplot(333)
# Axis for plotting 3D points
scatter = ax.scatter([], [], [], c='r', marker='o')
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_xlim(-1e1, 1e1)
ax.set_ylim(-1e1, 1e1)
ax.set_zlim(-1e1, 1e1)
ax.grid(True)
# Axis for 2D plotting camera views
for ax2d in [ax2d_cam1, ax2d_cam2]:
    ax2d.set_aspect('equal')
    ax2d.set_xlabel('X [pixels]')
    ax2d.set_ylabel('Y [pixels]')
    ax2d.grid(True)
scatter2d_cam1 = ax2d_cam1.scatter([], [], c='r', marker='o')
scatter2d_cam2 = ax2d_cam2.scatter([], [], c='r', marker='o')
plt.show()  # Show the initial plot

# Aruco dictionary being used for each camera
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
# Camera matrix and distortion coefficients
K = np.float32([[1.23637547e+03, 0.00000000e+00, 2.94357998e+02],  # IMX335
                [0.00000000e+00, 1.22549758e+03, 2.20521015e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
d = np.float32([-1.5209742, 2.46986782, 0.07634431, 0.07220847, 1.9630171])
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(2)

# Cost function reprojection
@staticmethod
def cost_func(var, pts0, pts1, K, d, n_points, n_cameras):
    # Extract X, R, and t from var
    X = var[:n_points*3].reshape(n_points, 3)
    rvec = var[n_points*3 : n_points*3+3].reshape(3,1)
    tvec = var[-3 : ].reshape(3,1)
    # Project 3D points to 2D for each camera
    proj_cam0, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, d)
    proj_cam1, _ = cv2.projectPoints(X, rvec, tvec, K, d)
    proj_cam0 = proj_cam0.reshape(-1, 2)
    proj_cam1 = proj_cam1.reshape(-1, 2)
    # Error in reprojection
    err = np.concatenate([proj_cam0 - pts0, proj_cam1 - pts1]).ravel()
    return err

n_points = 16
n_cameras = 2
while True:
    ############################################################
    # Feature Detection
    ############################################################
    _, image0 = cam0.read()
    _, image1 = cam1.read()
    # Undistort both images
    image0 = cv2.undistort(image0, K, d)
    image1 = cv2.undistort(image1, K, d)
    # Run aruco detection
    pts0, _, _ = detector.detectMarkers(image0)
    pts1, _, _ = detector.detectMarkers(image1)
    pts0 = np.array(pts0)
    pts1 = np.array(pts1)
    # pts0 and pts1 are 16x2 arrays of aruco marker corners in 2D image
    if (len(pts0) == 4) and (len(pts0) == len(pts1)):
        pts0 = np.concatenate([arr.reshape(-1, 2) for arr in pts0])
        pts1 = np.concatenate([arr.reshape(-1, 2) for arr in pts1])
    else:
        print('Not enough markers detected!', end='\r')
        continue

    ############################################################
    # Optimization
    ############################################################
    n_points = len(pts0)
    var = np.zeros(n_points*3+6)
#    print('Original Cost = {}'.format(np.linalg.norm(cost_func(var, pts0, pts1, K, d, n_points, 2))))
    # Optimize X, R and t to minimize cost_func
    # meausure time taken for optimisation
    start_time = time.time()
    res = least_squares(cost_func, var, method='lm',
                        args=(pts0, pts1, K, d, n_points, n_cameras))
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Cost = {}'.format(np.linalg.norm(res.fun)))
    ############################################################
    ############################################################
    # If any values in res are NaN or Inf, continue
    if np.isnan(res.x).any() or np.isinf(res.x).any():
        continue
    # Recollect X, R, t from res
    X = res.x[:n_points * 3].reshape(n_points,3)
    rvec = res.x[n_points * 3:n_points * 3 + 3].reshape(3,1)
    tvec = res.x[n_points * 3 + 3:n_points * 3 + 6].reshape(3,1)
    var = res.x
    print(var.reshape(-1, 3))

    # Add 2D points from cam1 and cam2 to the plot
    ax2d_cam1.set_xlim(min(pts0[:, 0]), max(pts0[:, 0]))
    ax2d_cam1.set_ylim(min(pts0[:, 1]), max(pts0[:, 1]))
    scatter2d_cam1.set_offsets(pts0)
    ax2d_cam2.set_xlim(min(pts1[:, 0]), max(pts1[:, 0]))
    ax2d_cam2.set_ylim(min(pts1[:, 1]), max(pts1[:, 1]))
    scatter2d_cam2.set_offsets(pts1)
    # Add 3D points triangulated to the plot
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0, 1)
    scatter._offsets3d = (X[:,0], X[:,1], X[:,2])  # Update data points
    plt.draw()  # Redraw the plot
    plt.pause(0.005)
