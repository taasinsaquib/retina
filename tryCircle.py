import numpy as np
import open3d as o3d

import time

from helpers_general      import makeCircleXY, vecAngle
from helpers_o3d          import setupScene, rayCast
from helpers_retina       import visualizeHits
from helpers_deepLearning import loadModel, convertONV, onvToNN

def meat(translate, scene, pcd, line_set, octree, rays, nRays, pupil, lastOnv, lastCenter, seeLines, seeHits, model):

    pcd.translate(translate)

    greyKernel = np.array([0.299, 0.587, 0.114])

    # system to move the sphere across the screen
    x = translate[0]
    y = translate[1]

    polX = polY = 1
    if x < 0:
        polX = -1

    if y < 0:
        polY = -1

    octree.clear()
    octree.convert_from_point_cloud(pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine
    
    onv, hits, searchRay = rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits)
    curOnv = np.sum(onv * greyKernel, axis=1)
    
    curCenter = pcd.get_center()
    binaryDeltaOnv = convertONV(curOnv, lastOnv)

    # inference with nn
    if model is not None:

        output = onvToNN(model, binaryDeltaOnv)
        print(output, translate)   # for now, labels are how much the pcd was translated
                                                # TODO: experiment with deltaGaze and center of pcd

        # TODO: feed NN angle outputs to move retina (have to collect data first)
        # calculate shift in gaze

        r = vecAngle(pupil, lastCenter, curCenter, polX, polY)

        # rotate retina points
        linePoints = np.asanyarray(line_set.points)
        rays = rays @ r.T
        for i in range(0, nRays):
            linePoints[i*2] = rays[i]
        line_set.points = o3d.utility.Vector3dVector(linePoints)
        # TODO: do a new raycast from this location?
        # TODO: saccades?

    return curOnv, binaryDeltaOnv


def collectData(rays, pupil, radius, seeLines=False, seeHits=False, seeDistribution=False, saveData=False, model=None, nPoints=10000, w=200, h=200):

    nRays = len(rays)
    pcd, scene, line_set = setupScene(seeLines, nPoints, w, h, rays, nRays)

    greyKernel = np.array([0.299, 0.587, 0.114])

    # create octree
    octree = o3d.geometry.Octree(max_depth=4)                   # > 4 makes search return empty later

    points = makeCircleXY(radius)

    data = []
    labels = []
    centers = []
    nZeros = 0

    lastOnv = None
    curOnv  = None
    binaryDeltaOnv = None

    lastCenter = None
    curCenter  = None

    # initialize the variables the first time
    octree.convert_from_point_cloud(pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine

    onv, hits, searchRay = rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits)
    lastOnv = np.sum(onv * greyKernel, axis=1)  # data
    lastCenter = pcd.get_center()               # labels

    centerOnv = lastOnv

    for x, y in points:

        # Move AWAY FROM center ****************************************
        
        # raycast, collect data and label
        translate = [x, y, 0]

        curOnv, binaryDeltaOnv = meat(translate, scene, pcd, line_set, octree, rays, nRays, pupil, lastOnv, lastCenter, seeLines, seeHits, model)

        # save data and label
        if saveData:
            data.append(binaryDeltaOnv)

            if np.count_nonzero(searchRay) > nRays - 10:
                labels.append([0, 0, 0])
                nZeros += 1
                centers.append([0, 0, 0])
            else:
                labels.append(translate)
                print(pcd.get_center())
                centers.append(pcd.get_center())

        if seeDistribution:
            visualizeHits(rays, hits, searchRay, binaryDeltaOnv, type='events')

        lastOnv = curOnv
        lastCenter = curCenter

        # input("hi")

        # Move BACK TO center ****************************************
        # new raycast with ball at center (actually just re-use the one at the start)
        # or collect data for movement towards...

        translate = [-x, -y, 0]
        pcd.translate(translate)
    
        # update graphics loop
        scene.update_geometry(pcd)
        if seeLines:
            scene.update_geometry(line_set)

        scene.poll_events()
        scene.update_renderer()

        # curOnv, binaryDeltaOnv = meat(translate, scene, pcd, line_set, octree, rays, nRays, pupil, lastOnv, lastCenter, seeLines, seeHits, model)

        # # save data and label
        # if saveData:
        #     data.append(binaryDeltaOnv)

        #     if np.count_nonzero(searchRay) > nRays - 10:
        #         labels.append([0, 0, 0])
        #         nZeros += 1
        #         centers.append([0, 0, 0])
        #     else:
        #         labels.append(translate)
        #         print(pcd.get_center())
        #         centers.append(pcd.get_center())

        curOnv = centerOnv
        binaryDeltaOnv *= -1


        if seeDistribution:
            visualizeHits(rays, hits, searchRay, binaryDeltaOnv, type='events')


        lastOnv = curOnv
        lastCenter = curCenter

        # input("hi2")

    scene.destroy_window()

    if saveData:
        data = np.array(data)
        labels  = np.array(labels)

        np.save(f'data/data_circle_away_{radius}',     data)
        np.save(f'data/labels_circle_away_{radius}',   labels)
        np.save(f'data/centers__circle_away_{radius}', centers)

        print(f'#Zero labels: {nZeros}, Data: {data.shape}, Labels: {labels.shape}')


def main():

    # Load retina distribution, shift along z-axis
    retina = np.load('./data/retina_dist.npy')
    retina = retina[:14400]
    retina[:, 2] += 0.5

    # experimental rays
    rays = np.array([
        [0, 0, 2],
        [0.1, 0, 2],
        [-0.1, 0, 2],
        [0, 0.1, 2],
        [0, -0.1, 2],
        [0.3, 0, 2],
        [-0.3, 0, 2],
        [0, 0.5, 2],
        [0, -0.5, 2],
        [0.75, 0, 2],
        [-0.75, 0, 2],
        [0, 0.75, 2],
        [0, -0.75, 2],
    ])

    # location of eye pinhole
    pupil = np.array([0, 0, 0])

    radii = [1]

    # load model
    m = None
    # m = loadModel('./models/onv_resnet_v1_dict')

    for r in radii:
        t1 = time.perf_counter()
        
        collectData(retina, pupil, r, seeLines=True, seeHits=False, seeDistribution=True, saveData=False, model=m, w=600, h=600)

        t2 = time.perf_counter()
        
        print(f'Minutes to complete {r} - { (t2-t1)/60 }')

    collectData(r=1)
    

if __name__ == "__main__":
    main()