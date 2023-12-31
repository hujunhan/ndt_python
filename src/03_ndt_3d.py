import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import pickle
import os
import time
import scipy

# Parameters
outlier_ratio = 0.55
resolution = 1.0
gauss_c1 = 10.0 * (1 - outlier_ratio)
gauss_c2 = outlier_ratio / np.power(resolution, 3)
gauss_d3 = -np.log(gauss_c2)
gauss_d1 = -np.log(gauss_c1 + gauss_c2) - gauss_d3
gauss_d2 = -2 * np.log(
    (-np.log(gauss_c1 * np.exp(-0.5) + gauss_c2) - gauss_d3) / gauss_d1
)


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
        if len(points) < 5:
            # delete the voxel with less than 5 points
            voxel_dict.pop(voxel_index)
            continue
        mean = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        cov_inv = np.linalg.inv(cov)
        if points.shape[0] == 1:
            cov = (
                np.eye(3) * cov
            )  # If only one point in voxel, use the variance as the diagonal of a 3x3 covariance matrix
        voxel_gaussians[voxel_index] = (mean, cov_inv)

    return voxel_gaussians


def get_c_s_from_x(x):
    """return regulated cos and sin value of x

    Args:
        x (_type_): _description_
    """
    if abs(x) < 1e-5:
        return 1.0, 0.0
    else:
        return np.cos(x), np.sin(x)


def first_deriv(pose, point):
    """the jacobian would be the sum of the jacobian of each point

    Args:
        pose (_type_): 6x1 array
        point (_type_): Nx3 array
    """
    x, y, z = pose[3], pose[4], pose[5]
    cx, sx = get_c_s_from_x(x)
    cy, sy = get_c_s_from_x(y)
    cz, sz = get_c_s_from_x(z)
    para_mat = np.zeros((8, 3))
    para_mat[0, :] = [(-sx * sz + cx * sy * cz), (-sx * cz - cx * sy * sz), (-cx * cy)]
    para_mat[1, :] = [(cx * sz + sx * sy * cz), (cx * cz - sx * sy * sz), (-sx * cy)]
    para_mat[2, :] = [(-sy * cz), sy * sz, cy]
    para_mat[3, :] = [sx * cy * cz, (-sx * cy * sz), sx * sy]
    para_mat[4, :] = [(-cx * cy * cz), cx * cy * sz, (-cx * sy)]
    para_mat[5, :] = [(-cy * sz), (-cy * cz), 0]
    para_mat[6, :] = [(cx * cz - sx * sy * sz), (-cx * sz - sx * sy * cz), 0]
    para_mat[7, :] = [(sx * cz + cx * sy * sz), (cx * sy * cz - sx * sz), 0]
    jac_all = np.matmul(para_mat, point.T)
    jac_mean = np.mean(jac_all, axis=1)
    # print(f"jac_mean: {jac_mean}")
    J_e = np.zeros((3, 6))
    J_e[:, 0:3] = np.eye(3)
    J_e[1, 3] = jac_mean[0]
    J_e[2, 3] = jac_mean[1]
    c = 2
    for i in [4, 5]:
        for j in [0, 1, 2]:
            J_e[j, i] = jac_mean[c]
            c += 1
    return J_e


def second_deriv(pose, point):
    """the hessian would be the sum of the hessian of each point

    Args:
        pose (_type_): _description_
    """
    x, y, z = pose[3], pose[4], pose[5]
    cx, sx = get_c_s_from_x(x)
    cy, sy = get_c_s_from_x(y)
    cz, sz = get_c_s_from_x(z)
    para_mat = np.zeros((15, 3))
    para_mat[0, :] = [(-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), sx * cy]
    para_mat[1, :] = [(-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), (-cx * cy)]
    para_mat[2, :] = [(cx * cy * cz), (-cx * cy * sz), (cx * sy)]
    para_mat[3, :] = [(sx * cy * cz), (-sx * cy * sz), (sx * sy)]
    para_mat[4, :] = [(-sx * cz - cx * sy * sz), (sx * sz - cx * sy * cz), 0]
    para_mat[5, :] = [(cx * cz - sx * sy * sz), (-sx * sy * cz - cx * sz), 0]
    para_mat[6, :] = [(-cy * cz), (cy * sz), (-sy)]
    para_mat[7, :] = [(-sx * sy * cz), (sx * sy * sz), (sx * cy)]
    para_mat[8, :] = [(cx * sy * cz), (-cx * sy * sz), (-cx * cy)]
    para_mat[9, :] = [(sy * sz), (sy * cz), 0]
    para_mat[10, :] = [(-sx * cy * sz), (-sx * cy * cz), 0]
    para_mat[11, :] = [(cx * cy * sz), (cx * cy * cz), 0]
    para_mat[12, :] = [(-cy * cz), (cy * sz), 0]
    para_mat[13, :] = [(-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), 0]
    para_mat[14, :] = [(-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), 0]
    hessian_all = np.matmul(para_mat, point.T)
    hessian_mean = np.mean(hessian_all, axis=1)
    H_e = np.zeros((18, 6))
    H_e[10:12, 3] = hessian_mean[0:2]
    H_e[13:15, 3] = hessian_mean[2:4]
    H_e[16:18, 3] = hessian_mean[4:6]
    H_e[10:12, 4] = hessian_mean[2:4]
    H_e[12:15, 4] = hessian_mean[6:9]
    H_e[15:18, 4] = hessian_mean[9:12]
    H_e[10:12, 5] = hessian_mean[4:6]
    H_e[12:15, 5] = hessian_mean[9:12]
    H_e[15:18, 5] = hessian_mean[12:15]
    H_e = H_e.reshape(6, 6, 3)
    # print(f"hessian_mean: \n{H_e}")
    return H_e


def objective_func(x, source, target_gauss, voxel_size):
    t1 = time.time()
    r = R.from_euler("zyx", x[3:6], degrees=False)
    rot_mat = r.as_matrix()
    transformed_points = source @ rot_mat.T + x[:3]
    cost = 0.0
    in_voxel_points = []
    g = np.zeros((6, 1))
    h = np.zeros((6, 6))
    for p in transformed_points:
        voxel_index = tuple(np.floor(p / voxel_size).astype(int))
        if voxel_index in target_gauss:
            in_voxel_points.append(p)
            mean, cov_inv = target_gauss[voxel_index]
            first_d = first_deriv(x, p)
            second_d = second_deriv(x, p)
            if np.linalg.matrix_rank(cov_inv) < 3:
                continue
            ## Update g and H in equation 6.11 and 6.12
            x_trans = p - mean
            e_x_cov_x = np.exp(-gauss_d2 * x_trans @ cov_inv @ x_trans)  # reuseable
            if e_x_cov_x > 1 or e_x_cov_x < 0:  ## error cheching
                continue
            cost += -gauss_d1 * e_x_cov_x  # cost equation 6.9
            e_x_cov_x = e_x_cov_x * gauss_d2 * gauss_d1
            for i in range(6):
                cov_dxd_pi = cov_inv @ first_d[i]
                g[i] += x_trans @ cov_dxd_pi * e_x_cov_x
                for j in range(6):
                    h[i, j] += (
                        e_x_cov_x
                        * (
                            -gauss_d1
                            * x_trans
                            @ cov_dxd_pi
                            * x_trans
                            @ (cov_inv * first_d[j])
                        )
                        + x_trans
                        @ (cov_inv * second_d[3 * i : 3 * (i + 1), j])
                        * e_x_cov_x
                        + first_d[j] @ cov_dxd_pi
                    )
    in_voxel_points = np.asarray(in_voxel_points)
    jac_val = first_deriv(x, in_voxel_points)
    hessian_val = second_deriv(x, in_voxel_points)
    print(hessian_val)
    hessian_inv = np.linalg.pinv(hessian_val)
    res = np.matmul(hessian_inv, jac_val.reshape((18, 1)))
    # res = np.linalg.lstsq(hessian_val, jac_val.reshape((18, 1)), rcond=None)[0]

    print(f"res: {res}")
    t2 = time.time()
    print(f"cost: {cost}, time: {t2-t1}")
    # print(f"x: {x}")
    # print(f"jac: {jac_val}")
    return res


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
    init_guess = np.asarray([1, 1, 0, 0, 0, 0])  # x, y, z, roll, pitch, yaw
    res = np.zeros((6, 1))
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # 2. Score Computation
    # 3. Jacobian Computation
    # 4. Optimization, manually update the transformation, calculate the jac and hessian analytically
    for _ in range(10):
        res = objective_func(init_guess, source, target_gauss, voxel_size)
        init_guess = init_guess + res.reshape((6,))

    # 5. Transformation Update


if __name__ == "__main__":
    # Read the data from the file.
    source_pcd = o3d.io.read_point_cloud(
        "data/1.pcd"
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

    ndt(source_pt, target_pt, 1.0)
