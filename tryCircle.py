import random
import numpy as np
import open3d as o3d

import time

from helpers_general      import makeCircleXY, vecAngle, generateRandPoint
from helpers_o3d          import setupScene, rayCast
from helpers_retina       import visualizeHits
from helpers_deepLearning import loadFC, convertONV, FC1toNN, convertONVDiff

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

    r, angles = vecAngle(pupil, lastCenter, curCenter, polX, polY)

    # inference with nn
    if model is not None:

        output = FC1toNN(model, binaryDeltaOnv)
        print(output, angles * 180/np.pi)   # for now, labels are how much the pcd was translated
                                                # TODO: experiment with deltaGaze and center of pcd

        # TODO: feed NN angle outputs to move retina (have to collect data first)
        # calculate shift in gaze

        # rotate retina points
        linePoints = np.asanyarray(line_set.points)
        rays = rays @ r.T
        for i in range(0, nRays):
            linePoints[i*2] = rays[i]
        line_set.points = o3d.utility.Vector3dVector(linePoints)
    
        # TODO: do a new raycast from this location?
        # TODO: saccades?

    return curOnv, binaryDeltaOnv, angles


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
    angles = []
    nZeros = 0

    lastOnv = None
    curOnv  = None
    binaryDeltaOnv = None

    lastCenter = None
    curCenter  = None
    centerCenter = None

    curAngles = None
    ogRays = rays       # points get rotated around

    # initialize the variables the first time
    octree.convert_from_point_cloud(pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine

    onv, hits, searchRay = rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits)
    lastOnv = np.sum(onv * greyKernel, axis=1)  # data
    lastCenter = pcd.get_center()               # labels

    centerCenter = lastCenter
    centerOnv = lastOnv

    # points = zip([1, 0, -1, 0], [0, 1, 0, -1])
    points = zip([0.2, 0.5, 1, 1.5, 2, 2.5], np.zeros(6))

    for x, y in points:

        # Move AWAY FROM center ****************************************

        # raycast, collect data and label
        translate = [x, y, 0]

        pcd.translate(translate)

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

        r, curAngles = vecAngle(pupil, lastCenter, curCenter, polX, polY)

        # inference with nn
        if model is not None:

            output = FC1toNN(model, binaryDeltaOnv)

            print(output, curAngles)   # for now, labels are how much the pcd was translated
                                                    # TODO: experiment with deltaGaze and center of pcd

            # TODO: feed NN angle outputs to move retina (have to collect data first)
            # calculate shift in gaze

            # rotate retina points
            """
            linePoints = np.asanyarray(line_set.points)
            rotatedRays = rays @ r.T
            for i in range(0, nRays):
                linePoints[i*2] = rotatedRays[i]
            line_set.points = o3d.utility.Vector3dVector(linePoints)

            if seeLines:
                scene.update_geometry(line_set)

            scene.poll_events()
            scene.update_renderer()
            """
            # TODO: do a new raycast from this location?
            # TODO: saccades?

            # curOnv, binaryDeltaOnv, curAngles = meat(translate, scene, pcd, line_set, octree, rays, nRays, pupil, lastOnv, lastCenter, seeLines, seeHits, model)

            # print(curAngles)

        # save data and label
        elif saveData:
            data.append(binaryDeltaOnv)

            if np.count_nonzero(searchRay) > nRays - 10:
                labels.append([0, 0, 0])
                nZeros += 1
                centers.append([0, 0, 0])
                angles.append([0, 0, 0])
            else:
                labels.append(translate)
                centers.append(pcd.get_center())
                angles.append(curAngles)

        if seeDistribution:
            visualizeHits(ogRays, hits, searchRay, binaryDeltaOnv, type='events')

        lastOnv = curOnv
        lastCenter = curCenter

        # Move BACK TO center ****************************************
        # new raycast with ball at center (actually just re-use the one at the start)
        # or collect data for movement towards...

        position2 = generateRandPoint(2.5, 2.5, 5)

        translate = [-x, -y, 0]
        pcd.translate(translate)

        # rotate retina points back
        """
        linePoints = np.asanyarray(line_set.points)
        for i in range(0, nRays):
            linePoints[i*2] = rays[i]
        line_set.points = o3d.utility.Vector3dVector(linePoints)

        if seeLines:
            scene.update_geometry(line_set)
        
        scene.update_geometry(pcd)
        scene.poll_events()
        scene.update_renderer()
        """

        curOnv = centerOnv
        curCenter = centerCenter

        """
        binaryDeltaOnv *= -1

        # save data and label
        if saveData:
            data.append(binaryDeltaOnv)
        
            if np.count_nonzero(searchRay) > nRays - 10:
                labels.append([0, 0, 0])
                nZeros += 1
                centers.append([0, 0, 0])
                angles.append([0, 0, 0])
            else:
                labels.append(translate)
                centers.append(pcd.get_center())
                angles.append([-1*curAngles[0], -1*curAngles[1], -1*curAngles[2]])

        if seeDistribution:
            visualizeHits(rays, hits, searchRay, binaryDeltaOnv, type='events')
        """
        lastOnv = curOnv
        lastCenter = curCenter

        pcd.center = [0, 0, 0]
        scene.update_geometry(pcd)
        scene.poll_events()
        scene.update_renderer()

    scene.destroy_window()

    if saveData:
        data = np.array(data)
        labels  = np.array(labels)

        np.save(f'data/data_circle_away_{radius}',    data)
        np.save(f'data/labels_circle_away_{radius}',  labels)
        np.save(f'data/centers_circle_away_{radius}', centers)
        np.save(f'data/angles_circle_away_{radius}',  angles)

        print(f'#Zero labels: {nZeros}, Data: {data.shape}, Labels: {labels.shape}')


def randomPoints(rays, pupil, radius, seeLines=False, seeHits=False, seeDistribution=False, saveData=False, model=None, nPoints=10000, w=200, h=200):
    nRays = len(rays)
    pcd, scene, line_set = setupScene(seeLines, nPoints, w, h, rays, nRays)

    greyKernel = np.array([0.299, 0.587, 0.114])

    # create octree
    octree = o3d.geometry.Octree(max_depth=4)                   # > 4 makes search return empty later

    points = makeCircleXY(radius)

    data = []
    angles = []
    nZeros = 0

    lastOnv = None
    curOnv  = None
    binaryDeltaOnv = None

    lastCenter = None
    curCenter  = None

    curAngles = None
    ogRays = rays       # points get rotated around

    # initialize the variables the first time
    octree.convert_from_point_cloud(pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine

    onv, hits, searchRay = rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits)
    lastOnv = np.sum(onv * greyKernel, axis=1)  # data
    lastCenter = pcd.get_center()               # labels

    # scene.run()

    for i in range(1000):

        position1 = generateRandPoint(2.5, 2.5, -10)
        curCenter = pcd.get_center()

        # Move AWAY FROM center ****************************************

        # raycast, collect data and label
        translate = position1 - np.array(curCenter)
        # print(position1, curCenter, translate)
        pcd.translate(translate)

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
        
        binaryDeltaOnv = convertONVDiff(curOnv, lastOnv)

        r, curAngles = vecAngle(pupil, lastCenter, curCenter, polX, polY)

        skip = False

        # inference with nn
        if model is not None:

            output = FC1toNN(model, binaryDeltaOnv)

            print(output, curAngles)   # for now, labels are how much the pcd was translated
                                                    # TODO: experiment with deltaGaze and center of pcd

            # TODO: feed NN angle outputs to move retina (have to collect data first)
            # calculate shift in gaze

            # rotate retina points
            """
            linePoints = np.asanyarray(line_set.points)
            rotatedRays = rays @ r.T
            for i in range(0, nRays):
                linePoints[i*2] = rotatedRays[i]
            line_set.points = o3d.utility.Vector3dVector(linePoints)

            if seeLines:
                scene.update_geometry(line_set)

            scene.poll_events()
            scene.update_renderer()
            """
            # TODO: do a new raycast from this location?
            # TODO: saccades?

            # curOnv, binaryDeltaOnv, curAngles = meat(translate, scene, pcd, line_set, octree, rays, nRays, pupil, lastOnv, lastCenter, seeLines, seeHits, model)

            # print(curAngles)

        
        # save data and label
        elif saveData:

            if np.count_nonzero(searchRay) > nRays - 10:
                skip = True                
            else:
                data.append(binaryDeltaOnv)
                angles.append(curAngles[:2])

        if seeDistribution:
            visualizeHits(ogRays, hits, searchRay, binaryDeltaOnv, type='events')

        if skip == False:
            lastOnv = curOnv
            lastCenter = curCenter

    scene.destroy_window()

    if saveData:
        data = np.array(data)
        angles  = np.array(angles)

        np.save(f'data/data_rand',    data)
        np.save(f'data/angles_rand',  angles)

        print(f'#Zero labels: {nZeros}, Data: {data.shape}, Labels: {angles.shape}')

import matplotlib.pyplot as plt

# good until 2.2
def main():

    # Look at data
    angles = np.load('./data/angles_rand.npy') * 180/np.pi
    plt.title('scatter')
    plt.scatter(angles[:, 0:1], angles[:, 1:2], marker='.')
    # plt.xlim([-0.35, 0.35])
    # plt.ylim([-0.35, 0.35])
    plt.show()
    print(np.load('./data/data_rand.npy')[:1][:200])
    input("hi")


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

    # radii = np.arange(0.2, 2, 0.1)
    # radii = np.arange(2, 3, 0.1)
    radii = [2, 2, 2]
    # load model
    m = None
    # m = loadFC('./models/fc_away_v1_dict')

    for r in radii:

        r = np.around(r, decimals=1)

        t1 = time.perf_counter()

        # collectData(retina, pupil, r, seeLines=False, seeHits=False, seeDistribution=False, saveData=False, model=m, w=600, h=600)
        randomPoints(retina, pupil, r, seeLines=False, seeHits=False, seeDistribution=False, saveData=True, model=m, w=600, h=600)

        t2 = time.perf_counter()
        
        print(f'*** Minutes to complete {r} - { (t2-t1)/60 } ***')


if __name__ == "__main__":
    main()

# TODO: add noise to data?
# TODO: run nengo model on ONV