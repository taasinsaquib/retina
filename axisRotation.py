import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R

from tryCasting import vecAngle

np.set_printoptions(suppress=True)

w = 600
h = 600

# create sphere
sphereColor = [0.1, 0.1, 0.7]
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color(sphereColor)

mesh_sphere.translate([0, 0, -2])

# set up scene with pcd
scene = o3d.visualization.VisualizerWithKeyCallback()
scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=100, visible=True)
scene.add_geometry(mesh_sphere)

sceneControl = scene.get_view_control()
sceneControl.set_zoom(1.5)

axes = ( [1, 0, 0], [0, 1, 0], [0, 0, -1] )

points = [
    [0, 0, 0],
    axes[0],
    axes[1],
    axes[2],
    mesh_sphere.get_center()
]

lines = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4]
]

colors = [[1, 0, 0] for i in range(len(lines))]
colors[3] = [0, 0, 1]

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

scene.add_geometry(line_set)

dx = 0.01
dy = 0.00

delta = [dx, dy, 0]

polx = poly = 1
direction = 'R'
if dx < 0:
    polx = -1
    direction = 'L'
if dy < 0:
    poly = -1

prevCenter = None
curCenter  = None

pupil = [0, 0, 0]

def followSphere():
    while True:
        mesh_sphere.translate(delta)
        curCenter = mesh_sphere.get_center()

        if prevCenter is not None:
            x, y, z = vecAngle(pupil, axes, prevCenter, curCenter)
            
            print(x * 180/np.pi, y * 180/np.pi, z * 180/np.pi)

            print(poly*z, polx*y, x)
            r = R.from_euler('xyz', [poly*z, polx*-y, x])
            
            ax = np.array(axes) @ np.array(r.as_matrix()).T

            points[1] = ax[0]
            points[2] = ax[1]
            # points[3] = ax[2]

            v1 = curCenter - pupil
            v1 = v1 / np.linalg.norm(v1)
            points[3] = v1

            points[4] = curCenter

            # rotate around cur Axes
            line_set.points = o3d.utility.Vector3dVector(points)

            colors = [[0, 1, 0] for i in range(len(lines))]
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # rotate axes

        prevCenter = curCenter

        scene.update_geometry(mesh_sphere)
        scene.update_geometry(line_set)

        scene.poll_events()
        scene.update_renderer()

        # input("hi")

r = 1

# x^2 + y^2 = r^2

x = np.arange(-r, r+0.1, 0.1)
y = np.sqrt(r**2 - np.square(x))    # plus/minus

x = np.hstack((x, x))
y = np.hstack((y, -1*y))

points = zip(x, y)

for px, py in points:
    mesh_sphere.translate([px, py, 0])

    scene.update_geometry(mesh_sphere)
    scene.poll_events()
    scene.update_renderer()

    """
        raycast, collect data and label
    """

    input("hi")
    mesh_sphere.translate([-px, -py, 0])

    scene.update_geometry(mesh_sphere)
    scene.poll_events()
    scene.update_renderer()

    """
        reset prev stuff
        raycast
    """

    input("hi2")