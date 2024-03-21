import numpy as np
import cv2
from copy import deepcopy
import time

n_cameras = 3
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
    video_captures.append(cam)  # Capture initial frames
    if len(video_captures) == n_cameras:
        break

_, image0 = video_captures[0].read()
video_captures[0].release()
_, image1 = video_captures[1].read()
video_captures[1].release()
_, image2 = video_captures[2].read()
video_captures[2].release()

# Print if images are none
print(image0 is None, image1 is None, image2 is None)
