import numpy as np
import cv2 as cv
import time

def getPosition():
    video_file = '/dev/video0'
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), "Camera Stream Error"
#    K = np.float32([[492.05018569,   0.,         322.97035084], # Webcam
#                    [  0.,         490.41305419, 250.543017  ],
#                    [  0.,           0.,           1.        ]])
#    d = np.float32([ 0.03998296, -0.06070568,  0.00857903, -0.00289493,  0.09078684])

    K = np.float32([[1.23637547e+03, 0.00000000e+00, 2.94357998e+02], # Robocam
                    [0.00000000e+00, 1.22549758e+03, 2.20521015e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    d = np.float32([-1.5209742,   2.46986782,  0.07634431,  0.07220847,  1.9630171 ])

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    aruco_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

    t = np.zeros(3)
    alpha = 0.9

    start_time = time.time()
    while time.time() - start_time < 5:
        valid, image = video.read()
        if not valid:
            break

        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray_image)

        robs, tobs = np.zeros(3), np.zeros(3)
        if ids is not None:
            for i in range(len(corners)):
                _, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.03, K, d)

                tobs += tvec.flatten()

            tobs /= len(corners)
            t = alpha*t + (1-alpha)*tobs

        #info = f'Rotation : {np.around(r, decimals=2)} :: Position : {np.around(t, decimals=2)}'

    video.release()

    t[2] = 0.3942*t[2] + 0.02 # Z position Calibrated for RoboCam
    return t
