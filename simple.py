from math import inf
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from retina import loadRetinaDistribution, calculateRayDirections, visualizeRetina

help(o3d.geometry.Octree)
help(o3d.geometry.OctreeInternalNode)
help(o3d.geometry.OctreeColorLeafNode)


# Create scene and ball
scene = o3d.t.geometry.RaycastingScene()

ball = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
ball.compute_vertex_normals()
ball.paint_uniform_color([0.1, 0.1, 0.7])
ball = o3d.t.geometry.TriangleMesh.from_legacy_triangle_mesh(ball)

# TODO: put sphere in scene
ball_id = scene.add_triangles(ball)

# cast rays
center = np.array([0, 0, 3])

retina_dist = loadRetinaDistribution('retina_dist.npy')
print(np.min(retina_dist), np.max(retina_dist))
directions = calculateRayDirections(retina_dist, center)

# plt.plot(retina_dist[:, 0:1], retina_dist[:, 1:2], '.')
# plt.xlim([-0.35, 0.35])
# plt.ylim([-0.35, 0.35])
# plt.show()

n = retina_dist.shape[0] - 1

rays = np.hstack((retina_dist[:n], directions))

rays = rays.astype(np.float32)

# print(rays.shape)

ans = scene.cast_rays(rays)

# print(ans.keys())
# print(ans.keys())
# help(o3d.t.geometry.RaycastingScene())

# hits = color of sphere (no KD tree stuff yet)

hits = ans['t_hit'].numpy()
print(hits.shape)

for i, h in enumerate(hits):

    c = 'b'

    # print(h)

    if h != inf:
        c = 'g'

    plt.plot(retina_dist[i, 0:1], retina_dist[i, 1:2], f'.{c}')

plt.xlim([-0.35, 0.35])
plt.ylim([-0.35, 0.35])
plt.show()

# animate ball
# figure out loop with RaycastScene, update geometry?

# store delta hits from last frame to make events
# save to image to see events

# put retina distribution on a surface, not a circle


# use hit_t to calculate x, y, z of intersection
# get distance from eye to ball 