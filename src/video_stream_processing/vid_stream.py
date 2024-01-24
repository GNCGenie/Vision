import numpy as np
import cv2
from copy import deepcopy
import time

cam = cv2.VideoCapture(0)
# Set exposure to manual mode
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cam.set(cv2.CAP_PROP_EXPOSURE , 0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Set image format to MJPEG
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# Set image format to YUV
#cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

_, image = cam.read()
# Print details about the captured image
print('Backend being used for capture:', cam.getBackendName())
print('Image size:', image.shape)
print('Image type:', image.dtype)
print('Image channels:', image.ndim)

while True:
    _, image = cam.read()
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Carry out auto exposure and contrast
#    image = cv2.equalizeHist(image)
    # Show the processed image
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
