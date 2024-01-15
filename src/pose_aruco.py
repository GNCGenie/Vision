import numpy as np
import cv2 as cv

video_file = '/dev/video0'
video = cv.VideoCapture(video_file)
assert video.isOpened(), "Camera Stream Error"
K = np.float32([[492.05018569,   0.,         322.97035084],
                [  0.,         490.41305419, 250.543017  ],
                [  0.,           0.,           1.        ]])
d = np.float32([ 0.03998296, -0.06070568,  0.00857903, -0.00289493,  0.09078684])

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
aruco_params = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

r, t = np.zeros(3), np.zeros(3)
center = np.zeros(2)
alpha = 0.9
normal = np.array([1,0,0])

while True:
    valid, image = video.read()
    if not valid:
        break

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray_image)

    if ids is not None:
        tobs = np.zeros(3)
        rMatObs = np.zeros((3,3))
        for i in range(len(corners)):
            rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.07, K, d)

            tobs += tvec.flatten()
            rotMat, _ = cv.Rodrigues(rvec)
            rMatObs += rotMat

        tobs /= len(corners)
        rMatObs /= len(corners)
        robs, _  = cv.Rodrigues(rMatObs)
        robs = robs.flatten()

        r = alpha*r + (1-alpha)*robs
        t = alpha*t + (1-alpha)*tobs

        centers = [np.mean(x[0], axis=0) for x in corners]
        center = alpha*center + (1-alpha)*np.mean(centers, axis=0)
        center = center.astype(int)

    info = f'Rotation : {np.around(r, decimals=2)} :: Position : {np.around(t, decimals=2)}'
    print(info)

video.release()
cv.destroyAllWindows()

