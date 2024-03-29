import numpy as np
import cv2
from copy import deepcopy
import time

def load_cameras(n_cameras):
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
#        cam.set(cv2.CAP_PROP_FPS, 30)
        video_captures.append(cam)
        if len(video_captures) == n_cameras:
            break
    return video_captures

def get_image(video_captures, release=False):
    disp=[]
    for i in range(n_cameras):
        _, image = video_captures[i].read()
        if release:
            video_captures[i].release()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (512, 384))
        disp.append(image)
    return disp

import concurrent.futures
def get_image_parallel(video_captures, release=False):
    n_cameras = len(video_captures)
    disp=[]
    def process_image(i):
        _, image = video_captures[i].read()
        if release:
            video_captures[i].release()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (512, 384))
        return image
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, i) for i in range(n_cameras)]
        for i, future in enumerate(futures):
            disp.append(future.result())
    return disp

n_cameras = 3
video_captures = load_cameras(n_cameras)
release = False
while True:
    # Measure time taken to read frame
    time_start = time.time()
    if release:
        video_captures = load_cameras(n_cameras)
    disp = get_image_parallel(video_captures, release)
#    disp = get_image(video_captures, release)
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
