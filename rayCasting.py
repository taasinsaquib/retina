import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import copy
import math
import pickle

import open3d as o3d
# import open3d_tutorial as o3dtut

# Load mesh and convert to open3d.t.geometry.TriangleMesh
cube = o3d.geometry.TriangleMesh.create_box().translate([0, 0, 0])
cube = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(cube)

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

print("HIIIIIII")
colors = mesh_sphere.vertex_colors
# for c in colors:
#     print(c)
# print(dir(cube.triangles))
# items = cube.triangles.items()
# for i in items:
    # print(i)

ts = np.asarray(cube.triangles)
print(ts)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
cube_id = scene.add_triangles(cube)

# We create two rays:
# The first ray starts at (0.5,0.5,10) and has direction (0,0,-1).
# The second ray start at (-1,-1,-1) and has direction (0,0,-1).
# rays = o3d.core.Tensor([[0.5, 0.5, 10, 0, 0, -1], [-1, -1, -1, 0, 0, -1]],
                    #    dtype=o3d.core.Dtype.Float32)

# print(dir(scene))

# rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
#     fov_deg=90,
#     center=[0, 0, 2],
#     eye=[2, 3, 0],
#     up=[0, 1, 0],
#     width_px=640,
#     height_px=480,
# )

# ans = scene.cast_rays(rays)

# plt.imshow(ans['t_hit'].numpy(), vmax=3)

# print(ans.keys())
# print(ans)

# help(o3d.t.geometry.RaycastingScene)
