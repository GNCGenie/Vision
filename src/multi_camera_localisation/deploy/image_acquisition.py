import cv2
import numpy as np

import concurrent.futures
def get_points_parallel(n_cameras, n_markers, detector, video_captures):
    active_cameras = 0
    pts = np.zeros((n_markers, 2, n_cameras))

    def process_image(i):
        nonlocal active_cameras
        _, image = video_captures[i]()
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Aruco marker detection
        pts_i, _, _ = detector.detectMarkers(image)
        if (pts_i is not None and len(pts_i) == n_markers):
            pts_i = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
            active_cameras += 1
            pts[:, :, i] = np.concatenate([arr.reshape(-1, 2) for arr in pts_i])
            return pts_i
        else:
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, i) for i in range(n_cameras)]
        for i, future in enumerate(futures):
            result = future.result()

    return pts, active_cameras

def init_cameras(n_cameras):
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
            print(f'Connected to {video_captures} cameras')
            break

        return video_captures

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

if __name__ == '__main__':
    n_cameras = 3
    video_captures = init_cameras(n_cameras)
    pts, active_cameras = get_points_parallel(n_cameras, 2, detector, video_captures)
