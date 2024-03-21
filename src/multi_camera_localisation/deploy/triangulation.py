import numpy as np
import cv2

def project(points_3d, rvec, tvec, K):
    ############################################################
    # Projection 3D points to 2D for each camera
    proj_points = Rotation.from_rotvec(rvec).apply(points_3d)
    proj_points += tvec
    proj_points = proj_points @ K.T
    proj_points /= proj_points[2, np.newaxis]
    return proj_points[:, :2]

def cost_func(var, rvecs, tvecs, pts, K, d, n_points, n_cameras):
    ############################################################
    # Cost function reprojection
    X = var[:n_points*3].reshape(n_points, 3)

    err = np.zeros((n_points*2, n_cameras))
    for i in range(n_cameras):
        proj = cv2.projectPoints(X, rvecs[i], tvecs[i], K, d)[0].reshape(-1,2)
        err[:,i] = (proj-pts[:,:,i]).ravel()

    return err.ravel()

def triangulate(pts, K, d, rvecs, tvecs):
    ############################################################
    # Triangulate points first for rough estimate
    n_cameras = rvecs.shape[0]
    start_time = time.time()
    P0 = K @ np.vstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0])).T
    P1 = K @ np.vstack((cv2.Rodrigues(rvecs[1])[0], tvecs[1])).T
    X = cv2.triangulatePoints(P0, P1, pts[:,:,0].T, pts[:,:,1].T)
    X /= X[3]
    X = X[:3].T
    print("Time to triangulate: %s" % (time.time() - start_time))

    return X

def bundle_adjustment(X, rvecs, tvecs, pts, K, d, n_points, n_cameras):
    ############################################################
    # Bundle Adjustment
    start_time = time.time()
    var = np.concatenate([X.ravel()])
    solution = least_squares(cost_func, var, args=(rvecs, tvecs, pts, K, d, n_points, n_cameras),
                             ftol=1e-15, xtol=1e-15, gtol=1e-15,
                             max_nfev=1000)
    cost = np.linalg.norm(solution.fun)
    print('Cost = {}'.format(cost))
    print("Time to optimize: %s" % (time.time() - start_time))
    optimized_vars = solution.x
    if np.isnan(optimized_vars).any() or np.isinf(optimized_vars).any():
        return var # If any values in res are NaN or Inf, continue
    else:
        return optimized_vars
