import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import copy
import math
import pickle

import open3d as o3d

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

# print("Convert mesh to a point cloud and estimate dimensions")
pcd = mesh_sphere.sample_points_poisson_disk(2000)
diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

# o3d.visualization.draw_geometries([mesh_sphere, pcd])

"""
# print("Define parameters used for hidden_point_removal")
camera = [0, 0, diameter]
radius = diameter * 100

# print("Get all points that are visible from given view point")
_, pt_map = pcd.hidden_point_removal(camera, radius)

# print("Visualize result")
pcd = pcd.select_by_index(pt_map)
# o3d.visualization.draw_geometries([pcd])
"""

tree = o3d.geometry.KDTreeFlann()
tree.set_geometry(mesh_sphere)

print("Paint the 1500th point red.")
pcd.colors[1500] = [0, 1, 0]

print(pcd.points[1500])

# print("Find its 200 nearest neighbors, and paint them blue.")
[k, idx, _] = tree.search_radius_vector_3d([0, 0, 0], 100)
np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]

o3d.visualization.draw_geometries([pcd])