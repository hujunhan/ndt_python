import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import pickle
import os
import time


def voxelization(point_cloud, voxel_size):
    # Shift point_cloud to positive quadrant
    # point_cloud -= np.min(point_cloud, axis=0)

    # 1. put the points into the grid/cells
    # Compute voxel indices
    voxel_indices = np.floor(point_cloud / voxel_size).astype(int)
    # Group points by voxel index
    voxel_dict = {}
    for point, voxel_index in zip(point_cloud, voxel_indices):
        voxel_index_t = tuple(
            voxel_index
        )  # Convert np.array to tuple to use it as dictionary key
        if voxel_index_t not in voxel_dict:
            voxel_dict[voxel_index_t] = []
        voxel_dict[voxel_index_t].append(point)

    # 2. calculate the mean and covariance matrix for each cell
    voxel_gaussians = {}
    for voxel_index, points in voxel_dict.items():
        points = np.asarray(points)
        mean = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        if points.shape[0] == 1:
            cov = (
                np.eye(3) * cov
            )  # If only one point in voxel, use the variance as the diagonal of a 3x3 covariance matrix
        voxel_gaussians[voxel_index] = (mean, cov)

    return voxel_gaussians


def objective_func(x, source, target_gauss, voxel_size=0.1):
    t1 = time.time()
    r = R.from_euler("zyx", x[3:6], degrees=False)
    rot_mat = r.as_matrix()
    transformed_points = source @ rot_mat.T + x[:3]
    cost = 0.0
    for p in transformed_points:
        voxel_index = tuple(np.floor(p / voxel_size).astype(int))
        if voxel_index in target_gauss:
            mean, cov = target_gauss[voxel_index]
            if np.linalg.matrix_rank(cov) < 3:
                continue
            cost += (p - mean) @ np.linalg.inv(cov) @ (p - mean)
    t2 = time.time()
    print(f"cost: {cost}, time: {t2-t1}")
    print(f"x: {x}")
    return cost


def ndt(source, target, voxel_size=0.1):
    # 1. Initialization. Grid Construction / Voxelization
    ## tyically, the voxel size is 1/10 of the map size
    ## only the target map is voxelized
    t1 = time.time()
    if not os.path.exists("./dump/target_gauss.pkl"):
        target_gauss = voxelization(target, voxel_size)
        pickle.dump(target_gauss, open("./dump/target_gauss.pkl", "wb"))
    else:
        target_gauss = pickle.load(open("./dump/target_gauss.pkl", "rb"))
    t2 = time.time()
    print("voxelization time: ", t2 - t1)
    # Registration
    init_guess = np.zeros(6)  # x, y, z, roll, pitch, yaw
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # 2. Score Computation
    # 3. Jacobian Computation
    # 4. Optimization
    res = minimize(
        objective_func,
        init_guess,
        args=(source, target_gauss, voxel_size),
        method="BFGS",
        options={"disp": True, "maxiter": 20},
        tol=200,
    )
    print(res.x)
    # 5. Transformation Update


if __name__ == "__main__":
    # Read the data from the file.
    source_pcd = o3d.io.read_point_cloud(
        "data/2.pcd"
    )  # generally, the source is the current frame
    target_pcd = o3d.io.read_point_cloud(
        "data/map0606.pcd"
    )  # generally, the target is the map
    # o3d.visualization.draw_geometries(geometry_list=[target_pcd])

    # get the numpy array, the shape is (N,3)
    source_pt = np.asarray(source_pcd.points)
    target_pt = np.asarray(target_pcd.points)

    # # add 1 to the last column, shape is (N,4), for homogeneous transformation
    # source_pt = np.hstack((source_pt, np.ones((source_pt.shape[0], 1))))
    # target_pt = np.hstack((target_pt, np.ones((target_pt.shape[0], 1))))

    ndt(source_pt, target_pt)
