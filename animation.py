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

np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold = False) 

scene = o3d.visualization.Visualizer()
scene.create_window(window_name='Main Scene', width=200, height=200, left=200, top=500, visible=True)
scene.add_geometry(mesh_sphere)

# lastImage = o3d.geometry.Image()
lastImage = None
curDisplay = o3d.geometry.Image()

eyeVis = o3d.visualization.Visualizer()
eyeVis.create_window(window_name='Left Eye', width=200, height=200, left=200, top=700, visible=True)
eyeVis.add_geometry(curDisplay)

# print(type(mesh_sphere), type(curDisplay))

# TODO: make black and white events to start

for i in range(0, 10):
    mesh_sphere.translate((0.01, 0, 0))

    scene.update_geometry(mesh_sphere)
    scene.poll_events()
    scene.update_renderer()
    
    curImage = scene.capture_screen_float_buffer(do_render=True)

    # test = np.asarray(curImage)
    # print(np.sum(test != 1.))
    # plt.imshow(test)
    
    # scene.capture_screen_image(f'/Users/Saquib/Desktop/{i}.png', do_render=True)

    # display the difference between the frames
    if lastImage != None:
        events = np.asarray(curImage) - np.asarray(lastImage)
        
        print(np.count_nonzero(events < 0), np.count_nonzero(events > 0), np.count_nonzero(events))
        curDisplay = o3d.geometry.Image(events)

        eyeVis.update_geometry(curDisplay)
        eyeVis.poll_events()
        eyeVis.update_renderer()
        # o3d.io.write_image(f'/Users/Saquib/Desktop/{i}.png', curImage)

    lastImage = curImage

scene.destroy_window()
eyeVis.destroy_window()