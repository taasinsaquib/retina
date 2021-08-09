import numpy as np
import open3d as o3d

import math

# Animating a sphere in this example

mesh_sphere = None

def moveLeft(vis):
    mesh_sphere.translate((-0.1, 0, 0))

def moveRight(vis):
    mesh_sphere.translate((0.1, 0, 0))

def moveUp(vis):
    mesh_sphere.translate((0, 0.1, 0))

def moveDown(vis):
    mesh_sphere.translate((0, -0.1, 0))

# for horizontal motion
def generateTrainingDataH(eyeCenter, delta, collectData=False, w=200, h=200, c=3):

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
    scene = o3d.visualization.VisualizerWithKeyCallback()
    scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=500, visible=True)
    scene.add_geometry(mesh_sphere)

    scene.register_key_callback(ord('A'), moveLeft)
    scene.register_key_callback(ord('D'), moveRight)
    scene.register_key_callback(ord('W'), moveUp)
    scene.register_key_callback(ord('S'), moveDown)

    sceneControl = scene.get_view_control()
    sceneControl.set_zoom(1.5)

    lastImage  = None
    lastCenter = None
    curDisplay = o3d.geometry.Image()

    dataSet = []
    labels = []

    # figure out how many steps to take based on dx
    distX = 1.5 - (-1.5) + 2 * 0.25      # maxX - minX + 2 * radius of sphere + little extra
    nStepsX = math.ceil(distX / delta * pol) # ceil to make sure x is always reset correctly after the inner loop runs

    # keep y movement constant to get screen coverage
    nStepsY = 36

    # mesh_sphere.translate((-delta * nStepsX/2, -0.1 * nStepsY/2, 0))

    print(nStepsY, nStepsX)

    nZeros = 0

    for i in range(0, int(nStepsY)):
        lastImage  = None
        lastCenter = None
        for j in range(0, int(nStepsX)):

            # mesh_sphere.translate((delta, 0, 0))
            # print(mesh_sphere.get_center())

            if j%5 == 0:
                sceneControl.camera_local_rotate(2.1, 0.0, 0.0, 0.0)
            elif j%5 == 4:
                sceneControl.camera_local_rotate(-2.1, 0.0, 0.0, 0.0)

            scene.update_geometry(mesh_sphere)
            scene.poll_events()
            scene.update_renderer()
            
            curImage = scene.capture_screen_float_buffer(do_render=True)
            curCenter = mesh_sphere.get_center()
            
            # scene.capture_screen_image(f'/Users/Saquib/Desktop/{i}.png', do_render=True)

            # display the difference between the frames
            if lastImage != None and collectData:

                events = np.asarray(curImage, dtype=np.int8) - np.asarray(lastImage, dtype=np.int8)

                # make binary events
                binEvents = np.zeros((w, h))

                bright = np.argwhere(events > 0)
                dim    = np.argwhere(events < 0)

                binEvents[bright[:, :2]] = 1
                binEvents[dim[:, :2]]    = -1

                # print(np.count_nonzero(events < 0), np.count_nonzero(events > 0), np.count_nonzero(events))
                # print(len(dim), len(bright))

                # for visualization purposes
                events[events>0] = 150

                # convert to b/w for binary events
                # bw = np.dot(events[...,:3], [0.299, 0.587, 0.114])

                curDisplay = o3d .geometry.Image(events)
                # o3d.io.write_image(f'/Users/Saquib/Desktop/{j}.png', curDisplay)

                # label
                [sX, sY, _] = curCenter
                if (abs(sX) > 1.5 + 0.25) or (abs(sY) > 1.5 + 0.25):    # deltaGaze zero'd out if sphere not in view
                    deltaGaze = [0, 0, 0]
                    print('z')
                    nZeros += 1
                else:
                    deltaGaze = curCenter - lastCenter

                print(deltaGaze)

                dataSet.append(binEvents)
                labels.append(deltaGaze)


            lastImage = curImage
            lastCenter = curCenter

        # mesh_sphere.translate((-nStepsX * delta, 0.1, 0))

    if collectData:
        print(f'Total Size: {len(dataSet)}, Zeros: {nZeros}')

        dataSet = np.array(dataSet)
        labels  = np.array(labels)

        # save data
        np.save(f'data/data_dist_{delta*pol}_{dir}', dataSet)
        np.save(f'data/labels_dist_{delta*pol}_{dir}', labels)

        print(nZeros, dataSet.shape, labels.shape)

    scene.destroy_window()

# rates = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
rates = [0.01]

eyeCenter = [0, 0, 3]

for r in rates:
    for i in [1, -1]:
        generateTrainingDataH(eyeCenter, i*r, False)


# create saccade
# Slayer, Bindsnet, Nengo
