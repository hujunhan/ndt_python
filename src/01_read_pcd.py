import open3d as o3d
import numpy as np
import copy

file_path = "./data/map0606.pcd"
# file_path='./data/map.pcd'
# file_path='/home/darkblue/code/oh_my_loam/data/nsh/1422133389.108526080.pcd'
pcd = o3d.io.read_point_cloud(file_path)

# downsample the point cloud
pcd = pcd.voxel_down_sample(voxel_size=0.2)
points = np.asarray(pcd.points)
points = points[points[:, 2] > 0.1]
pcd.points = o3d.utility.Vector3dVector(points)
print(f"mean: {np.mean(points, axis=0)}")
print(f"num of points: {points.shape[0]}")


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )


# visualize the point cloud from top view, z axis is up, bird view
o3d.visualization.draw_geometries(geometry_list=[pcd])

# compress to 2d plane by averaging the z axis
## make grid, 0.1m per grid
grid_size = 0.1
x_min = np.min(points[:, 0])
x_max = np.max(points[:, 0])
y_min = np.min(points[:, 1])
y_max = np.max(points[:, 1])
z_min = np.min(points[:, 2])
z_max = np.max(points[:, 2])
x_grid_num = int((x_max - x_min) / grid_size)
y_grid_num = int((y_max - y_min) / grid_size)
print(f"x_grid_num: {x_grid_num}, y_grid_num: {y_grid_num}")
print(
    f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}, z_min: {z_min}, z_max: {z_max}"
)

points = (points / grid_size).astype(np.int32)
print(f"num of points: {points.shape[0]}")
# average the z axis, compress to 2d plane
# if there are multiple points in one grid, average the z axis
# use the points[0:2] as the key of the dict
from collections import defaultdict

points_2d = defaultdict(float)
points_counter = defaultdict(int)
for p in points:
    points_2d[(p[0], p[1])] += p[2]
    points_counter[(p[0], p[1])] += 1

for p in points_2d:
    points_2d[p] = points_2d[p] / points_counter[p]

print(f"num of points_2d: {len(points_2d)}")

# convert to numpy array [n,3]
points_average = np.asarray([[p[0], p[1], points_2d[p]] for p in points_2d])
print(points_average.shape)

# filter the points by z axis, remove the points that are too high or too low
min_z = np.min(points_average[:, 2])
max_z = np.max(points_average[:, 2])
print(f"min_z: {min_z}, max_z: {max_z}")
points_average = points_average[points_average[:, 2] > 100]
# 3d plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

# ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
plt.scatter(points_average[:, 0], points_average[:, 1], s=1)
# equal ratio
plt.axis("equal")
plt.show()
