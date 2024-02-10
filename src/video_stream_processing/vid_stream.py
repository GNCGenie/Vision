import numpy as np
import cv2
from copy import deepcopy
import time

n_cameras = 4
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

    disp=[]
    for i in range(n_cameras):
        _, image = video_captures[i]()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (512, 384))
        disp.append(image)

    return disp

while True:
    # Measure time taken to read frame
    time_start = time.time()
    disp = getPoints()
    time_end = time.time()
    print('Time taken to read frame:', time_end - time_start, 'seconds')

    # Join all images in display
    disp = np.concatenate(disp, axis=1)
    cv2.imshow('Display', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

    # Carry out auto exposure and contrast
#    image = cv2.equalizeHist(image)

    #_, image = cam.read()
    ## Print details about the captured image
    #print('Backend being used for capture:', cam.getBackendName())
    #print('Image size:', image.shape)
    #print('Image type:', image.dtype)
    #print('Image channels:', image.ndim)
