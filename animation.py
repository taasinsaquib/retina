import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import copy
import math
import pickle

import open3d as o3d

# Animating a sphere in this example
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

eye = o3d.geometry.TriangleMesh.create_sphere(radius=.1)
eye.compute_vertex_normals()
eye.paint_uniform_color([0, 1, 1])

# ** Method 1 ** #

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

key_to_callback = {}
key_to_callback[ord("K")] = change_background_to_black

# o3d.visualization.draw_geometries_with_key_callbacks([mesh_sphere], key_to_callback)

# ** Method 2 ** #

# http://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html
def rotate_view(vis):
    # print(type(vis))
    # ctr = vis.get_view_control()
    # ctr.rotate(10.0, 0.0)

    mesh_sphere.translate((0.1, 0, 0))

    vis.update_geometry(mesh_sphere)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_float_buffer()
    return False

# o3d.visualization.draw_geometries_with_animation_callback([mesh_sphere],
#                                                             rotate_view)


# Create a new window

# eyeVis = o3d.visualization.Visualizer()
# eyeVis.create_window(window_name='Left Eye', width=100, height=100, left=100, top=600, visible=True)
# eyeVis.add_geometry(mesh_sphere)
# eyeVis.run()
# eyeVis.destroy_window()

# Two windows at once

# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold = False)

w = 200
h = 200
c = 3

eyeCenter = [0, 0, 3]
eye.translate(eyeCenter)

scene = o3d.visualization.Visualizer()
scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=500, visible=True)
scene.add_geometry(mesh_sphere)
# scene.add_geometry(eye)

sceneControl = scene.get_view_control()

# lastImage = o3d.geometry.Image()
lastImage = None
curDisplay = o3d.geometry.Image()

eyeVis = o3d.visualization.Visualizer()
eyeVis.create_window(window_name='Left Eye', width=w, height=h, left=200, top=700, visible=True)
eyeVis.add_geometry(curDisplay)

dataSet = []
labels = []

for i in range(0, 10):
    mesh_sphere.translate((0.1, 0, 0))

    # sceneControl.camera_local_translate(forward=0., right=0.1, up=0.)
    # sceneControl.camera_local_rotate(x=0.1, y=0.)

    # experiment with rotation
    # R = mesh_sphere.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    # mesh_sphere.rotate(R, center=(0, 0, 0))

    scene.update_geometry(mesh_sphere)
    scene.poll_events()
    scene.update_renderer()
    
    curImage = scene.capture_screen_float_buffer(do_render=True)
    
    # scene.capture_screen_image(f'/Users/Saquib/Desktop/{i}.png', do_render=True)

    # display the difference between the frames
    if lastImage != None:

        events = np.asarray(curImage, dtype=np.int8) - np.asarray(lastImage, dtype=np.int8)

        # make binary events
        binEvents = np.zeros((w, h))

        bright = np.argwhere(events > 0)
        dim    = np.argwhere(events < 0)

        binEvents[bright[:, :2]] = 1
        binEvents[dim[:, :2]]    = -1

        # label
        deltaGaze = mesh_sphere.get_center() - eyeCenter

        dataSet.append(binEvents)
        labels.append(deltaGaze)

        # print(np.count_nonzero(events < 0), np.count_nonzero(events > 0), np.count_nonzero(events))
        # print(len(dim), len(bright))

        # for visualization purposes
        events[events>0] = 150

        # convert to b/w for binary events
        bw = np.dot(events[...,:3], [0.299, 0.587, 0.114])

        curDisplay = o3d.geometry.Image(events)
        # o3d.io.write_image(f'/Users/Saquib/Desktop/{i}.png', curDisplay)

        eyeVis.update_geometry(curDisplay)
        eyeVis.poll_events()
        eyeVis.update_renderer()

    lastImage = curImage

# save data
np.save('data/data', dataSet)
np.save('data/labels', labels)

scene.destroy_window()
eyeVis.destroy_window()