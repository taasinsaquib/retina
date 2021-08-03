import numpy as np
import open3d as o3d

w = 200
h = 200

# create sphere
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

pcd = mesh_sphere.sample_points_poisson_disk(10000)

# create scene, add sphere to it
scene = o3d.visualization.VisualizerWithKeyCallback()
scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=500, visible=True)
scene.add_geometry(pcd)

sceneControl = scene.get_view_control()
sceneControl.set_zoom(1.5)

pcd.translate([0, 0, -3])

rays = np.array([
    [0, 0, 2],
    [0.5, 0, 2],
    [-0.5, 0, 2],
])

nRays = len(rays)

pupil = np.array([0, 0, 1.5])

# not sure if I should initialize to something else
# 0 should be at the ray origin
hits =  [0] * nRays

# visualization stuff
seeLines = True
points = np.zeros((2*nRays, 3))
lines = [[i, i+1] for i in range(0, nRays*2, 2)]
colors = [[1, 0, 0] for i in range(len(lines))]

for i in range(0, nRays):
    points[i*2] = rays[i]

# TODO: put this in the loop, update odd indices with current points
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

if seeLines:
    scene.add_geometry(line_set)

print(mesh_sphere.get_center())

for t in np.arange(0, 10, 1):

    # TODO: visualize rays

    # not sure if this updates the tree (probs not?)
    # pcd.translate([0.1, 0, 0])

    tree = o3d.geometry.KDTreeFlann()
    tree.set_geometry(pcd)

    curPoints = rays * (1-t) + pupil * t

    if seeLines:
        for i in range(0, nRays):
            line_set.points[2*i+1] = curPoints[i]

    # TODO: only search if there isn't a hit yet, or figure out a way to know there won't be a collision
    for i in range(nRays):
        cur = curPoints[i]

        # using this, can't traverse tree with API
        k, idx, _ = tree.search_hybrid_vector_3d(cur, 0.1, 1)
        print(cur, k, idx)
        
        if k > 0 and hits[i] == 0:
            np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]
            print("hit ", t)
            hits[i] = t

    scene.update_geometry(pcd)
    if seeLines:
        scene.update_geometry(line_set)

    scene.poll_events()
    scene.update_renderer()

    scene.run()

print(hits)

# scene.run()

scene.destroy_window()
