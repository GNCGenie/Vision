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
# Axis for plotting 3D points
scatter = ax.scatter([], [], [], c='r', marker='o')
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_zlim(-2.0, 2.0)
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

# Aruco dictionary being used for each camera
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
# Camera matrix and distortion coefficients
K = np.float32([[1.23637547e+03, 0.00000000e+00, 2.94357998e+02],  # IMX335
                [0.00000000e+00, 1.22549758e+03, 2.20521015e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
d = np.float32([-1.5209742, 2.46986782, 0.07634431, 0.07220847, 1.9630171])

# Cost function reprojection
@staticmethod
def cost_func(var, pts, K, d, n_points, n_cameras):
    X = var[:n_points*3].reshape(n_points, 3)
    rvecs = var[n_points*3:n_points*3+3*n_cameras].reshape(n_cameras, 3)
    tvecs = var[-3*n_cameras:].reshape(n_cameras, 3)

    rvecs[0] = tvecs[0] = np.zeros(3)
    err = np.zeros((n_points*2, n_cameras))
    for i in range(n_cameras):
        proj = cv2.projectPoints(X, rvecs[i], tvecs[i], K, d)[0].reshape(-1,2)
        err[:,i] = (proj-pts[:,:,i]).ravel()
    return err.ravel()

# Create lists to hold camera objects and video captures
n_markers = 4
n_points = 4*n_markers
n_cameras = 3  # Change this to the desired number of cameras
# Initialize cameras and video captures
video_captures = []
for i in range(0, 10):
    cam = cv2.VideoCapture(i)  # Adjust camera indices as needed
    # Capture cam reading error
    if not cam.isOpened():
        continue
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_EXPOSURE , 0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Set image format to MJPEG
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    video_captures.append(cam.read)  # Capture initial frames
    if len(video_captures) == n_cameras:
        break

def getPoints():
    ############################################################
    # Feature Detection
    active_cameras = 0
    for i in range(n_cameras):
        _, image = video_captures[i]()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Carry out auto exposure and contrast
        # image = cv2.equalizeHist(image)
        pts_i, _, _ = detector.detectMarkers(image)
        if (len(pts_i) == n_markers):
            pts_i = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
            active_cameras += 1
            pts[:, :, i] = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
        else:
            continue
    return pts, active_cameras

alpha = 0.1
pts = np.zeros((n_points, 2, n_cameras))
pts_prev = np.zeros((n_points, 2, n_cameras))
X = np.ones((n_points, 3))
rvecs = np.zeros((n_cameras, 3))
tvecs = np.zeros((n_cameras, 3))
while True:
    pts, active_cameras = getPoints()
    if active_cameras < 2:
        print('Not enough active cameras', end='\r')
        continue
    pts = alpha * pts + (1 - alpha) * pts_prev
    pts_prev = pts

    ############################################################
    # Optimization
    var = np.concatenate([X.ravel(), rvecs.ravel(), tvecs.ravel()])
    start_time = time.time()
    solution = least_squares(cost_func, var, method='trf',
                             args=(pts, K, d, n_points, n_cameras),
                             bounds=(-10, 10),
                             max_nfev=1000)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Cost = {}'.format(np.linalg.norm(solution.fun)))
    optimized_vars = solution.x
    if np.isnan(optimized_vars).any() or np.isinf(optimized_vars).any():
        continue # If any values in res are NaN or Inf, continue
    var = optimized_vars

    ############################################################
    # Recollect X, rvecs, tvecs from res
    X = var[:n_points * 3].reshape(n_points, 3)
    rvecs = var[n_points * 3:n_points * 3 + 3 * n_cameras].reshape(-1, 3)
    tvecs = var[n_points * 3 + 3 * n_cameras:].reshape(-1,3)
    print('Mean position of 3D points = {}'.format(np.mean(X, axis=0)))

    # Add 2D points from cam1 and cam2 to the plot
    for i,ax in enumerate([ax2d_cam1, ax2d_cam2, ax2d_cam3]):
        ax.set_xlim(min(pts[:, 0, i]), max(pts[:, 0, i]))
        ax.set_ylim(min(pts[:, 1, i]), max(pts[:, 1, i]))
        # Set new data
        scatter2d[i].set_offsets(pts[:, :, i])

    # Add 3D points triangulated to the plot
    scatter._offsets3d = (X[:,0], X[:,1], X[:,2])  # Update data points
    plt.draw()  # Redraw the plot
    plt.pause(0.001)
