import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import math
import time
import torch

from deepLearning import loadModel


def setupScene(seeLines, nPoints, w, h, rays, nRays):

    # create sphere
    sphereColor = [0.1, 0.1, 0.7]
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color(sphereColor)

    # point cloud (pcd) from mesh (to add to KDTree)
    pcd = mesh_sphere.sample_points_poisson_disk(nPoints)
    pcd.translate([0, 0, -3])

    # set up scene with pcd
    scene = o3d.visualization.VisualizerWithKeyCallback()
    scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=100, visible=True)
    scene.add_geometry(pcd)

    sceneControl = scene.get_view_control()
    sceneControl.set_zoom(1.5)

    line_set = None

    # code to visualize rays with a LineSet
    if seeLines:
        points = np.zeros((2*nRays, 3))
        lines = [[i, i+1] for i in range(0, nRays*2, 2)]
        colors = [[1, 0, 0] for i in range(len(lines))]

        # ray origin points at even indices, odd indices will be replaced with the current end of the ray
        for i in range(0, nRays):
            points[i*2] = rays[i]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )

        line_set.colors = o3d.utility.Vector3dVector(colors)
        scene.add_geometry(line_set)

    return pcd, scene, line_set


def octreeSearch(cur, octree, pcdPoints, seeHits):
    
    found = False
    idx = -1

    leaf, _ = octree.locate_leaf_node(cur)

    if leaf is not None:
        # print("HIT", octree.is_point_in_bound(cur, octree.origin, octree.size), cur, leaf.color, leaf.indices)
        
        # get L2 distance from current point to points in the leaf node
        candidates = pcdPoints[leaf.indices]
        dist = np.linalg.norm(candidates - cur, axis=1)

        # if any point is within 0.1, choose the closest point as the hit
        if np.count_nonzero(dist < 0.1) > 0:
            found = True

            # mark closest point as green
            if seeHits:
                closeIdx = np.argmin(dist)
                idx = leaf.indices[closeIdx]

    return found, idx


def rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits):
    
    # setup data structs to hold info
    hits =  [0] * nRays                     # hit distances of rays, 0 if no intersection
    searchRay = [True] * nRays              # store if corresponding ray index has hit yet
    onv = np.ones((nRays, 3))               # store colors of the ray hits, white (1, 1, 1) if not hit 

    for t in np.arange(0, 10, 0.1):

        pcdPoints = np.asanyarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # extend the rays
        curPoints = rays * (1-t) + pupil * t

        # only search rays who haven't hit yet
        indices = np.argwhere(searchRay)

        # set odd indices of the points array, as mentioned above
        if seeLines:
            for i in indices.flatten():
                line_set.points[2*i+1] = curPoints[i]

        for i in indices.flatten():
            cur = curPoints[i]
            hit, closeIdx = octreeSearch(cur, octree, pcdPoints, seeHits)

            if hit == True:
                onv[i] = colors[closeIdx]
                
                hits[i] = t
                searchRay[i] = False

                if seeHits:
                    colors[closeIdx, :] = [0, 1, 0]

        # update graphics loop
        scene.update_geometry(pcd)
        if seeLines:
            scene.update_geometry(line_set)

        scene.poll_events()
        scene.update_renderer()

    # print("Done")
    # input("Done!, press Enter to continue...")

    # just to experiment with where the sphere falls in the ray distribution
    # for i in range(0, 10):
    #     input("enter")
    #     pcd.translate([0.1, 0, 0])

    #     scene.update_geometry(pcd)
    #     scene.poll_events()
    #     scene.update_renderer()

    return onv, hits, searchRay

# plot ray hits and misses, or hit distances
def visualizeHits(rays, hits, searchRay, binaryHits, type='', distance=True):

    title = ''

    # set default background color for rays that didn't intersect w/ geometry
    color = ['dimgrey'] * len(hits)

    if type == 'events' and binaryHits is not None:
        title = 'Events'

        for i, s in enumerate(binaryHits):
            if s == -1:
                # color[i] = 0.5
                color[i] = 'dodgerblue'
            elif s == 1:
                # color[i] = 1
                color[i] = 'coral'

    elif type == 'distance':
        title = 'Distance to Surface'

        # visualize hit distance
        hits += np.min(hits)
        hits /= np.max(hits)

        color = hits

    else:
        title = 'Hit Locations'
        for i, s in enumerate(searchRay):
            if s == False:
                # color[i] = 1
                color[i] = 'coral'

    plt.title(title)
    plt.scatter(rays[:, 0:1], rays[:, 1:2], marker='.', c=color)
    plt.xlim([-0.35, 0.35])
    plt.ylim([-0.35, 0.35])
    plt.show()


# take the diff in greyscale values, reutrn a vector with {-1, 0, 1} aka events
def convertONV(curOnv, prevOnv):

    deltaOnv = curOnv - prevOnv

    negIdx  = np.argwhere(deltaOnv < 0)
    posIdx  = np.argwhere(deltaOnv > 0)
    zeroIdx = np.argwhere(deltaOnv == 0)

    # make deltaOnv into -1, 0, 1
    binaryDeltaOnv = deltaOnv
    binaryDeltaOnv[negIdx]  = -1
    binaryDeltaOnv[posIdx]  = 1
    binaryDeltaOnv[zeroIdx] = 0

    neg  = len(negIdx)
    pos  = len(posIdx)
    zero = len(zeroIdx)
    print(f'Zero: {zero}, Dim: {neg}, Bright: {pos}')

    return binaryDeltaOnv


def sphereRetinaRayCast(rays, pupil, delta, seeLines=False, seeHits=False, seeDistribution=False, saveData=False, model=None, nPoints=10000, w=200, h=200):

    nRays = len(rays)

    pcd, scene, line_set = setupScene(seeLines, nPoints, w, h, rays, nRays)

    # create octree
    octree = o3d.geometry.Octree(max_depth=4)                   # > 4 makes search return empty later

    greyKernel = np.array([0.299, 0.587, 0.114])

    # system to move the sphere across the screen
    moveLeft = [delta, 0, 0]
    nZeros = 0

    pol = 1
    direction = 'R'
    if delta < 0:
        pol = -1
        direction = 'L'

    # figure out how many steps to take based on dx
    distX = 3 - (-3) + 2 * 0.25      # maxX - minX + 2 * radius of sphere + little extra
    nStepsX = math.ceil(distX / delta * pol) # ceil to make sure x is always reset correctly after the inner loop runs

    # keep y movement constant to get screen coverage
    nStepsY = 15

    # pcd.translate((-delta * nStepsX/2, -0.1 * nStepsY/2, 0))
    pcd.translate((-delta * nStepsX//8, 0, 0))

    data = []
    labels = []
    centers = []

    for i in range(0, int(nStepsY)):
        lastOnv = None
        curOnv  = None
        binaryDeltaOnv = None

        for j in range(0, int(nStepsX)):
            pcd.translate(moveLeft)
            
            # octree.translate([0.1, 0, 0])     # Not implemented... so we have to keep clearing and adding in the geometry
            octree.clear()
            octree.convert_from_point_cloud(pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine
            
            onv, hits, searchRay = rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits)
            curOnv = np.sum(onv * greyKernel, axis=1)

            # process onv, save data and corresponding label
            if lastOnv is not None:
                binaryDeltaOnv = convertONV(curOnv, lastOnv)

                if model is not None:
                    squareOnv = np.reshape(binaryDeltaOnv, (120, 120))  # 14400 photoreceptors reshaped to a 120*120 square
                    channelEvents = np.zeros((1, 2, 120, 120))

                    bright = np.argwhere(squareOnv > 0)
                    dim    = np.argwhere(squareOnv < 0)

                    channelEvents[0][0][bright[:, :2]] = 1
                    channelEvents[0][1][dim[:, :2]]    = -1

                    input = torch.from_numpy(channelEvents)
                    with torch.no_grad():
                        output = model(input)
                    print(output.cpu().numpy(), moveLeft)   # for now, labels are how much the pcd was translated
                                                            # TODO: experiment with deltaGaze and center of pcd

                elif saveData:
                    data.append(binaryDeltaOnv)

                    if np.count_nonzero(searchRay) > nRays - 10:
                        labels.append([0, 0, 0])
                        nZeros += 1
                        centers.append([0, 0, 0])
                    else:
                        labels.append(moveLeft)
                        print(pcd.get_center())
                        centers.append(pcd.get_center())

            lastOnv = curOnv

            # print("# rays that missed: ", np.count_nonzero(searchRay))

            if seeDistribution:
                visualizeHits(rays, hits, searchRay, binaryDeltaOnv, type='events')
    
        pcd.translate((-nStepsX * delta, 0.1, 0))

    scene.destroy_window()

    if saveData:
        data = np.array(data)
        labels  = np.array(labels)

        np.save(f'data/data_dist_{delta*pol}_{direction}', data)
        np.save(f'data/labels_dist_{delta*pol}_{direction}', labels)
        np.save(f'data/centers_dist_{delta*pol}_{direction}', labels)

        print(f'#Zero labels: {nZeros}, Data: {data.shape}, Labels: {labels.shape}')


#*************************************************************#
# main()
#*************************************************************#

def main():

    # Load retina distribution, shift along z-axis
    retina = np.load('./data/retina_dist.npy')
    retina = retina[:14400]
    retina[:, 2] += 1

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
    pupil = np.array([0, 0, 0.5])

    # rates = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
    rates = [-0.4]

    # load model
    m = None
    m = loadModel('./models/onv_resnet_v1_dict')

    for r in rates:
        t1 = time.perf_counter()

        for i in [1, -1]:
            sphereRetinaRayCast(retina, pupil, r*i, seeLines=False, seeHits=False, seeDistribution=True, saveData=False, model=m, w=600, h=600)

        t2 = time.perf_counter()

        print(f'Minutes to complete {0.1} - {(t2-t1) / 60}')

if __name__=="__main__":
    main()