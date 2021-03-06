import numpy as np
import torch
from torch._C import device

from animation import moveLeft, moveRight, moveUp, moveDown
from helpers_deepLearning import loadModel

import open3d as o3d


eyeCenter = [0, 0, 3]

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
            # Jk aren't labels just however much the sphere moved?
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

        mesh_sphere.translate((-0.1, 0, 0))
        print("HI")

    scene.destroy_window()
    lineScene.destroy_window()

# MAIN

model = loadModel('./models/resnet_dist_v1_dict')



sceneWithNN(eyeCenter, model)

# sceneWithNN(eyeCenter)