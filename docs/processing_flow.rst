Processing Flow and Description
==============================

The localization of points in 3D space is addressed through two distinct sub-problems:

**1. Camera Calibration:**

- **Intrinsic parameter calibration:** This determines the internal characteristics of each camera (e.g., focal length, lens distortion). This is typically done beforehand and remains fixed for subsequent localization tasks.
- **Extrinsic parameter calibration:** This determines the position and orientation of each camera relative to a common world coordinate system. To achieve this, a known pattern (e.g., a checkerboard) is used to minimize the reprojection error between observed feature points and their predicted locations based on the estimated camera poses. The position of the calibration target defines the origin of the camera system.

**2. Point Localization:**

- **Rough guess:** An initial estimate of the 3D positions of points is obtained.
- **Bundle adjustment:** Multiple iterations of bundle adjustment are performed, refining the camera poses and 3D point positions simultaneously. This process leverages updated measurements of detected points, passed through an exponential average to minimize the impact of random noise.
- **Feature detection and initialization:** After fixing the extrinsic parameters, specific features (e.g., Aruco markers) to be localized are detected. Their initial 3D locations are approximated via triangulation.
- **Bundle optimization:** Bundle optimization is applied to the initial 3D point guess, minimizing the reprojection errors between all cameras, resulting in the final refined 3D point locations.
