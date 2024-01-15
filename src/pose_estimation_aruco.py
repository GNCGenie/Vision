import numpy as np
import cv2 as cv

video_file = '/dev/video0'
video = cv.VideoCapture(video_file)
assert video.isOpened(), "Camera Stream Error"
# K = np.array([[492.05018569,   0.,         322.97035084], # Webcam
#               [  0.,         490.41305419, 250.543017  ],
#               [  0.,           0.,           1.        ]])
# d = np.array([ 0.03998296, -0.06070568,  0.00857903, -0.00289493,  0.09078684])

K = np.array([[1032.15103813529, 0, 302.8362766815523],
              [0, 1027.323376124159, 348.4957147394717],
              [0, 0, 1]])
d = np.array([-2.88438811e-01,  1.05801596e-01, -1.03903198e-04, -1.00258193e-04, -1.95883202e-02])

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
arucoParams = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(arucoDict,arucoParams)

obj_file = '/home/Inspecity/Desktop/Aruco Marker Board.png'
image = cv.imread(obj_file)
(obj, ids, rejected) = detector.detectMarkers(image)
obj = [i[0] for i in obj]
obj = np.array(obj, dtype=np.float32)
obj = np.vstack(obj)
obj = np.hstack((obj, np.zeros((len(obj), 1), dtype=np.float32)))
obj = 5*obj/1e3

r,t = np.zeros(3),np.zeros(3)
center = np.zeros(2)
while True:
    valid,image = video.read()
    if not valid:
        break

    (img, ids, rejected) = detector.detectMarkers(image)
    if len(img)==4:
        img = np.vstack(img).reshape(-1, 1, 2).astype(np.float32)
        (ret, rvec, tvec) = cv.solvePnP(obj, img, K, d)
        tvec = tvec.flatten()

        alpha = 0.9
        t = (t*alpha+(1-alpha)*tvec)
        r = (r*alpha+(1-alpha)*rvec)

    print(t)
    info = f'Rotation : {r} :: Position : {t}'
    cv.putText(image, info,(10,50), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))

    cv.imshow('Aruco type detection', image)
    key = cv.waitKey(1)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break
