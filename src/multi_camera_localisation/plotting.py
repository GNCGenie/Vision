import matplotlib.pyplot as plt
############################################################
# Initialise plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Axis for plotting 3D points
scatter = ax.scatter([], [], [], c='r', marker='o')
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_zlim(-0.0, 2.0)
ax.grid(True)
# Axis for 2D plotting camera views
ax2d_cam1 = fig.add_subplot(331)
ax2d_cam2 = fig.add_subplot(333)
ax2d_cam3 = fig.add_subplot(337)
ax2d_cam4 = fig.add_subplot(339)
for ax2d in [ax2d_cam1, ax2d_cam2, ax2d_cam3, ax2d_cam4]:
    ax2d.set_aspect('equal')
    ax2d.set_xlabel('X [pixels]')
    ax2d.set_ylabel('Y [pixels]')
    ax2d.grid(True)
scatter2d_cam1 = ax2d_cam1.scatter([], [], c='r', marker='o')
scatter2d_cam2 = ax2d_cam2.scatter([], [], c='r', marker='o')
scatter2d_cam3 = ax2d_cam3.scatter([], [], c='r', marker='o')
scatter2d_cam4 = ax2d_cam4.scatter([], [], c='r', marker='o')
scatter2d = [scatter2d_cam1, scatter2d_cam2, scatter2d_cam3, scatter2d_cam4]
plt.show()  # Show the initial plot

############################################################
# Visualize the reconstructed 3D points
def visualize(pts, X):
    # Add 2D points from cam1 and cam2 to the plot
    n_cameras = len(pts[0, 0, :])
    n_points = len(pts[:, 0, 0])
    for i,ax in zip(range(n_cameras), [ax2d_cam1, ax2d_cam2, ax2d_cam3, ax2d_cam4]):
        ax.set_xlim(min(pts[:, 0, i]), max(pts[:, 0, i]))
        ax.set_ylim(min(pts[:, 1, i]), max(pts[:, 1, i]))
        # Set new data
        scatter2d[i].set_offsets(pts[:, :, i])

    # Add 3D points triangulated to the plot
    scatter._offsets3d = (X[:,0], X[:,1], X[:,2])
    plt.draw()  # Redraw the plot
    plt.pause(0.001)
