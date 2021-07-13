import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import copy
import math
import pickle

import open3d as o3d

# create noisy log polar offsets from center point
def generateRetinaDistribution(radius, alpha=360, rho=41, save=''):

    noiseOffsets = []

    mean = 0
    var = 0.0025
    stdDev = math.sqrt(var)

    for p in range(1, rho):
        mult = radius * (math.exp(p) / math.exp(rho-1))

        for a in range(0, alpha):
            direction = np.array([math.cos(a), math.sin(a), 0])

            rand1 = np.random.normal(loc=mean, scale=stdDev)
            rand2 = np.random.normal(loc=mean, scale=stdDev)
            
            noise = np.array([rand1, rand2, 0])

            offset = mult * direction + noise

            noiseOffsets.append([offset[0], offset[1], 0])

    # placeholder for the center, which is the last point
    noiseOffsets.append(np.zeros(3))

    noiseOffsets = np.array(noiseOffsets)

    if save != '':
        with open('retina_dist.npy', 'wb') as f:
            np.save(f, noiseOffsets)

    return noiseOffsets


def loadRetinaDistribution(f):
    return np.load(f)

def visualizeRetina(center, pinhole, offsets, alpha=360, rho=41):

    points = offsets + center
    n = len(points)

    points[-1] = pinhole  # replace the placeholder w/ location of pinhole

    # draw to the center for now
    lines = []
    for i in range(0, n-1):
        lines.append([i, n-1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    colors = []
    for i in range(0, n):
        if i < 1000:
            c = [1, 0, 0]
        elif 1000 < i and i < 2000:
            c = [0, 1, 0]
        else:
            c = [0, 0, 1]
        colors.append(c)

    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


"""
help(o3d.t.geometry)

# only needed for tutorial, monkey patches visualization
sys.path.append('..')
import open3d_tutorial
# change to True if you want to interact with the visualization windows
# open3d_tutorial.interactive = not "CI" in os.environ
open3d_tutorial.interactive = True

# Load mesh and convert to open3d.t.geometry.TriangleMesh
cube = o3d.geometry.TriangleMesh.create_box().translate([0, 0, 0])
cube = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(cube)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
cube_id = scene.add_triangles(cube)

print(cube_id)
"""

# """
print("Let's define some primitives")
mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
                                                height=1.0,
                                                depth=1.0)
mesh_box.compute_vertex_normals()
mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
                                                          height=4.0)
mesh_cylinder.compute_vertex_normals()
mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[-2, -2, -2])

eyeRadius = 0.2

# huh = o3d.geometry.Geometry2D.TriangleMesh.create_circle(radius = 0.2)
# help(o3d.geometry.Geometry2D)
# help(o3d.geometry.Geometry2D.TriangleMesh)

leftEye = o3d.geometry.TriangleMesh.create_sphere(radius=eyeRadius).translate((0, 0, 3))
leftEye.paint_uniform_color([0.1, 0.9, 0.1])
print(f'Center of mesh leftEye: {leftEye.get_center()}')

leftEyeConverge = leftEye.get_center()
leftEyeConverge[2] -= eyeRadius     # looking in the z direction when initialized
print(leftEyeConverge)

print("We draw a few primitives using collection.")
# o3d.visualization.draw_geometries([mesh_sphere, leftEye, mesh_frame])
center = np.array([0, 0, 3])

line_set = visualizeRetina(center, leftEyeConverge, generateRetinaDistribution(eyeRadius))
o3d.visualization.draw_geometries([leftEye, line_set, mesh_box])

# print("We draw a few primitives using + operator of mesh.")
# o3d.visualization.draw_geometries(
#     [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])
# """

"""
print("Let's draw a box using o3d.geometry.LineSet.")
points = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]
lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([line_set])
"""