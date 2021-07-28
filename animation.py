from pickle import load
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch._C import device
import torchvision
import math

from MyResnet import MyResNet

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
"""
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
"""

# for horizontal motion
def generateTrainingDataH(eyeCenter, delta, w=200, h=200, c=3):

    """
        pol (int) -1 or 1, if the sphere moves right to left or L or R
        eyeCenter ([float, float, float]) position of the center of the eye
        delta (float) how much the sphere moves in one step
    """

    # lineset to see boundaries of viewing frustum
    """
    points = [
        [1.5, 0, 0],
        [-1.5, 0, 0],
        [0, 1.5, 0],
        [0, -1.5, 0],
    ]

    lines = [
        [0, 1],
        [2, 3]
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    """

    pol = 1
    dir = 'R'
    if delta < 0:
        pol = -1
        dir = 'L'

    # create sphere
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

    # create scene, add sphere to it
    scene = o3d.visualization.Visualizer()
    scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=500, visible=True)
    scene.add_geometry(mesh_sphere)
    # scene.add_geometry(line_set)
    scene.register_key_callback(ord('K'), change_background_to_black)

    sceneControl = scene.get_view_control()
    # print(sceneControl.get_field_of_view())
    # sceneControl.set_constant_z_far(-2)
    # sceneControl.camera_local_translate(forward=0., right=0., up=0.)
    sceneControl.set_zoom(1.5)

    lastImage = None
    curDisplay = o3d.geometry.Image()

    dataSet = []
    labels = []

    # figure out how many steps to take based on dx
    distX = 1.5 - (-1.5) + 2 * 0.25      # maxX - minX + 2 * radius of sphere + little extra
    nStepsX = math.ceil(distX / delta * pol) # ceil to make sure x is always reset correctly after the inner loop runs

    # keep y movement constant to get screen coverage
    nStepsY = 36

    mesh_sphere.translate((-delta * nStepsX/2, -0.1 * nStepsY/2, 0))

    print(nStepsY, nStepsX)

    nZeros = 0

    for i in range(0, int(nStepsY)):
        for j in range(0, int(nStepsX)):

            # mesh_sphere.translate((delta, 0, 0))
            # print(mesh_sphere.get_center())

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
                [sX, sY, _] = mesh_sphere.get_center()
                if (abs(sX) > 1.5 + 0.25) or (abs(sY) > 1.5 + 0.25):    # deltaGaze zero'd out if sphere not in view
                    deltaGaze = [0, 0, 0]
                    nZeros += 1
                else:
                    deltaGaze = mesh_sphere.get_center() - eyeCenter

                dataSet.append(binEvents)
                labels.append(deltaGaze)

                # print(np.count_nonzero(events < 0), np.count_nonzero(events > 0), np.count_nonzero(events))
                # print(len(dim), len(bright))

                # for visualization purposes
                events[events>0] = 150

                # convert to b/w for binary events
                # bw = np.dot(events[...,:3], [0.299, 0.587, 0.114])

                curDisplay = o3d .geometry.Image(events)
                # o3d.io.write_image(f'/Users/Saquib/Desktop/{j}.png', curDisplay)

                # eyeVis.update_geometry(curDisplay)
                # eyeVis.poll_events()
                # eyeVis.update_renderer()

            lastImage = curImage

        # mesh_sphere.translate((-nStepsX * delta, 0.1, 0))

    dataSet = np.array(dataSet)
    labels  = np.array(labels)

    # save data
    # np.save(f'data/data_{delta*pol}_{dir}', dataSet)
    # np.save(f'data/labels_{delta*pol}_{dir}', labels)

    print(nZeros, dataSet.shape, labels.shape)

    scene.destroy_window()

# rates = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]
# rates = [0.1]

eyeCenter = [0, 0, 3]

# for r in rates:
#     for i in [1, -1]:
#         generateTrainingDataH(eyeCenter, i*r)


# Code to Run a NN and infer where the sphere center is

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadModel(path):
    modResnet = torchvision.models.resnet18(pretrained=True)
    modResnet.fc = nn.Linear(512, 3)

    model = MyResNet(modResnet)
    model.load_state_dict(torch.load(path, map_location=device))
    
    model.double()
    model.to(device)
    model.eval()

    return model

def moveLeft(vis):
    mesh_sphere.translate((-0.1, 0, 0))

def moveRight(vis):
    mesh_sphere.translate((0.1, 0, 0))

def moveUp(vis):
    mesh_sphere.translate((0, 0.1, 0))

def moveDown(vis):
    mesh_sphere.translate((0, -0.1, 0))

# create sphere
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

# create LineSet
points = [
    [0, 0, 0],
    [1, 0, 0],
]

lines = [
    [0, 1]
]

colors = [
    [1, 0, 0]
]

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

def createScenes(w, h, c):

    global mesh_sphere, line_set

    scene = o3d.visualization.VisualizerWithKeyCallback()
    scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=500, visible=True)
    scene.add_geometry(mesh_sphere)

    sceneControl = scene.get_view_control()
    sceneControl.set_zoom(1.5)

    # can't hold two together, just works the first time then picks one direction to continue in
    scene.register_key_callback(ord('A'), moveLeft)
    scene.register_key_callback(ord('D'), moveRight)
    scene.register_key_callback(ord('W'), moveUp)
    scene.register_key_callback(ord('S'), moveDown)

    lineScene = o3d.visualization.Visualizer() 
    lineScene.create_window(window_name='See Movement', width=w, height=h, left=200, top=700, visible=True)
    lineScene.add_geometry(line_set)

    lineControl = lineScene.get_view_control()
    lineControl.set_zoom(1.5)

    return scene, sceneControl, lineScene, lineControl

run = True

def sceneWithNN(eyeCenter, model=None, w=200, h=200, c=3):

    scene, _, lineScene, _ = createScenes(w, h, c)

    lastImage = None
    lastCenter = None

    while run == True:      # currently no way of stopping it, just hit ctrl-C
        scene.update_geometry(mesh_sphere)
        scene.poll_events()
        scene.update_renderer()

        lineScene.update_geometry(line_set)
        lineScene.poll_events()
        lineScene.update_renderer()

        curImage  = scene.capture_screen_float_buffer(do_render=True)
        curCenter = mesh_sphere.get_center()

        # display the difference between the frames
        if lastImage != None:

            events = np.asarray(curImage, dtype=np.int8) - np.asarray(lastImage, dtype=np.int8)

            # make binary events
            binEvents = np.zeros((w, h))

            bright = np.argwhere(events > 0)
            dim    = np.argwhere(events < 0)

            binEvents[bright[:, :2]] = 1
            binEvents[dim[:, :2]]    = -1

            # TODO: make binEvents 2 channel to input to NN
            channelEvents = np.zeros((1, 2, w, h), dtype=np.double)
            channelEvents[0][0][bright[:, :2]] = 1
            channelEvents[0][1][dim[:, :2]]    = -1

            # label
            # TODO: re-train model with labels being distance traveled
                # curCenter - lastCenter, not deltaGaze
            [sX, sY, _] = mesh_sphere.get_center()
            if (abs(sX) > 1.5 + 0.25) or (abs(sY) > 1.5 + 0.25):    # deltaGaze zero'd out if sphere not in view
                deltaGaze = [0, 0, 0]
            else:
                deltaGaze = mesh_sphere.get_center() - eyeCenter

            # run NN here
            if model != None:
                input = torch.from_numpy(channelEvents)
                with torch.no_grad():
                    output = model(input)
                print(output.cpu().numpy(), deltaGaze)

            # TODO: red dot where it thinks the center is using output

            # Draw line to show distance the center moved between frames
            # TODO: basically shows up as a dot because its so small
            line_set.points[0] = curCenter
            line_set.points[1] = lastCenter

        lastImage  = curImage
        lastCenter = curCenter

        mesh_sphere.translate((0.01, 0, 0))

    scene.destroy_window()
    lineScene.destroy_window()

# MAIN

# model = loadModel('./models/resnet_improvement_v3_dict')
# sceneWithNN(eyeCenter, model)

sceneWithNN(eyeCenter)