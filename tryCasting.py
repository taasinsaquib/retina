import numpy as np
import open3d as o3d

import math
import time

from helpers_general      import vecAngle
from helpers_o3d          import setupScene, rayCast
from helpers_retina       import visualizeHits
from helpers_deepLearning import loadModel, convertONV, onvToNN


def sphereRetinaRayCast(rays, pupil, translate, seeLines=False, seeHits=False, seeDistribution=False, saveData=False, model=None, nPoints=10000, w=200, h=200):

    nRays = len(rays)

    pcd, scene, line_set = setupScene(seeLines, nPoints, w, h, rays, nRays)

    # create octree
    octree = o3d.geometry.Octree(max_depth=4)                   # > 4 makes search return empty later

    greyKernel = np.array([0.299, 0.587, 0.114])

    # system to move the sphere across the screen
    dx = translate[0]
    dy = translate[1]
    nZeros = 0

    polX = polY = 1
    direction = 'R'
    if dx < 0:
        polX = -1
        direction = 'L'

    if dy < 0:
        polY = -1

    # figure out how many steps to take based on dx
    distX = 3 - (-3) + 2 * 0.25      # maxX - minX + 2 * radius of sphere + little extra
    nStepsX = math.ceil(distX / dx * polX) # ceil to make sure x is always reset correctly after the inner loop runs

    # keep y movement constant to get screen coverage
    nStepsY = 15

    # pcd.translate((-delta * nStepsX/2, -0.1 * nStepsY/2, 0))
    pcd.translate((-dx * nStepsX//8, 0, 0))

    data = []
    labels = []
    centers = []

    for i in range(0, int(nStepsY)):
        lastOnv = None
        curOnv  = None
        binaryDeltaOnv = None

        lastCenter = None
        curCenter  = None

        for j in range(0, int(nStepsX)):
            pcd.translate(translate)
            
            # octree.translate([0.1, 0, 0])     # Not implemented... so we have to keep clearing and adding in the geometry
            octree.clear()
            octree.convert_from_point_cloud(pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine
            
            onv, hits, searchRay = rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits)
            curOnv = np.sum(onv * greyKernel, axis=1)
            curCenter = pcd.get_center()

            # process onv, save data and corresponding label
            if lastOnv is not None:
                binaryDeltaOnv = convertONV(curOnv, lastOnv)
                
                # inference with nn
                if model is not None:

                    # output = onvToNN(model, binaryDeltaOnv)
                    # print(output, translate)   # for now, labels are how much the pcd was translated
                                                            # TODO: experiment with deltaGaze and center of pcd

                    # TODO: feed NN angle outputs to move retina (have to collect data first)
                    # calculate shift in gaze

                    r, angles = vecAngle(pupil, lastCenter, curCenter, polX, polY)

                    # rotate retina points
                    linePoints = np.asanyarray(line_set.points)
                    rays = rays @ r.T
                    for i in range(0, nRays):
                        linePoints[i*2] = rays[i]
                    line_set.points = o3d.utility.Vector3dVector(linePoints)
                    # TODO: do a new raycast from this location?
                    # TODO: saccades?

                # save data and label
                elif saveData:
                    data.append(binaryDeltaOnv)

                    if np.count_nonzero(searchRay) > nRays - 10:
                        labels.append([0, 0, 0])
                        nZeros += 1
                        centers.append([0, 0, 0])
                    else:
                        labels.append(translate)
                        print(pcd.get_center())
                        centers.append(pcd.get_center())

            lastOnv = curOnv
            lastCenter = curCenter

            # print("# rays that missed: ", np.count_nonzero(searchRay))

            if seeDistribution:
                visualizeHits(rays, hits, searchRay, binaryDeltaOnv, type='events')
    
        pcd.translate((-nStepsX * dx, 0.1, 0))

    scene.destroy_window()

    if saveData:
        data = np.array(data)
        labels  = np.array(labels)

        np.save(f'data/data_dist_{dx*polX}_{direction}', data)
        np.save(f'data/labels_dist_{dx*polX}_{direction}', labels)
        np.save(f'data/centers_dist_{dx*polX}_{direction}', centers)

        print(f'#Zero labels: {nZeros}, Data: {data.shape}, Labels: {labels.shape}')


#*************************************************************#
# main()
#*************************************************************#

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

    # rates = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
    rates = [0.1]

    # load model
    m = None
    # m = loadModel('./models/onv_resnet_v1_dict')

    for r in rates:
        t1 = time.perf_counter()

        for i in [1, -1]:
            sphereRetinaRayCast(retina, pupil, [r*i, 0, 0], seeLines=True, seeHits=False, seeDistribution=False, saveData=False, model=m, w=600, h=600)

        t2 = time.perf_counter()

        print(f'Minutes to complete {r} - {(t2-t1) / 60}')

if __name__=="__main__":
    main()

# TODO: collect data with sphere moving away from center of the retina
