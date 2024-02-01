import numpy as np
import cv2
from copy import deepcopy
import time

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Set image format to MJPG/YUYV
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # Set exposure to manual mode
cam.set(cv2.CAP_PROP_AUTO_WB, 1)
cam.set(cv2.CAP_PROP_EXPOSURE , 1e2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

_, image = cam.read()
# Print details about the captured image
print('Backend being used for capture:', cam.getBackendName())
print('Image size:', image.shape)
print('Image type:', image.dtype)
print('Image channels:', image.ndim)

while True:
    # Measure time taken to read frame
    time_start = time.time()
    _, image = cam.read()
    time_end = time.time()
#    print('Time taken to read frame:', time_end - time_start, 'seconds')
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Carry out auto exposure and contrast
#    image = cv2.equalizeHist(image)
    # Show the processed image
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
