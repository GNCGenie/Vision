import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
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
ax.set_zlim(-0.0, 1.0)
ax.grid(True)
# Axis for 2D plotting camera views
ax2d_cam1 = fig.add_subplot(331)
ax2d_cam2 = fig.add_subplot(333)
for ax2d in [ax2d_cam1, ax2d_cam2]:
    ax2d.set_aspect('equal')
    ax2d.set_xlabel('X [pixels]')
    ax2d.set_ylabel('Y [pixels]')
    ax2d.grid(True)
scatter2d_cam1 = ax2d_cam1.scatter([], [], c='r', marker='o')
scatter2d_cam2 = ax2d_cam2.scatter([], [], c='r', marker='o')
scatter2d = [scatter2d_cam1, scatter2d_cam2]
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
K = np.float32([[2008.103946092011, 0, 1282.248797977388], # IMX335 4K
                [0, 1994.058216315491, 1068.567637100975],
                [0, 0, 1]])
d = np.float32([-0.4509704449450611, 0.2490271613018018, -0.006054363779374283, -0.001358884013979639, -0.08083116341021042])
cam0 = cv2.VideoCapture(0)
cam0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cam0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cam0.set(cv2.CAP_PROP_EXPOSURE , 1e2)
cam0.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cam0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
cam1 = cv2.VideoCapture(2)
cam1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cam1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cam1.set(cv2.CAP_PROP_EXPOSURE , 1e2)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

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
X = np.ones((n_points, 3))
rvec = np.zeros((3,1))
tvec = np.zeros((3,1))
pts = np.zeros((n_points, 2, n_cameras))
while True:
    ############################################################
    # Feature Detection
    ############################################################
    _, image0 = cam0.read()
    _, image1 = cam1.read()
#    # Undistort both images
#    image0 = cv2.undistort(image0, K, d)
#    image1 = cv2.undistort(image1, K, d)
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
    pts[:,:,0] = pts0
    pts[:,:,1] = pts1

    ############################################################
    # Optimization
    ############################################################
    var = np.concatenate([X.ravel(), rvec.ravel(), tvec.ravel()])
#    var = np.zeros(var.shape)
#    print('Original Cost = {}'.format(np.linalg.norm(cost_func(var, pts0, pts1, K, d, n_points, 2))))
    # Optimize X, R and t to minimize cost_func
    # meausure time taken for optimisation
    start_time = time.time()
    res = least_squares(cost_func, var, method='trf',
                        ftol=1e-19, xtol=1e-19, gtol=1e-19,
                        bounds=(-10, 10), max_nfev=100,
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
    # Print mean position of 3D points
    print('Mean position of 3D points = {}'.format(np.mean(X, axis=0)))

############################################################
############################################################
############################################################
############################################################
    # Add 2D points from cam1 and cam2 to the plot
    for i,ax in enumerate([ax2d_cam1, ax2d_cam2]):
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
