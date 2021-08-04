import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


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
    scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=500, visible=True)
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
                # colors[leaf.indices[closeIdx], :] = [0, 1, 0]

    return found, idx


def rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits):
    
    # setup data structs to hold info
    hits =  [0] * nRays                     # hit distances of rays, 0 if no intersection
    searchRay = [True] * nRays              # store if corresponding ray index has hit yet
    onv = np.zeros((nRays, 3))               # store colors of the ray hits, black (0, 0, 0) if not hit 

    for t in np.arange(0, 10, 0.1):

        # TODO: translate octree as well
        # pcd.translate([0.1, 0, 0])
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
        # scene.run()

    print("Done!")
    scene.run()
    scene.destroy_window()

    return onv, hits, searchRay


def visualizeHits(rays, hits, searchRay, distance=True):
    # see hits on retina distribution
    
    # visualize hit distance
    hits += np.min(hits)
    hits /= np.max(hits)

    # just visualize if there was a hit or not
    color = np.zeros(len(hits))
    for i, s in enumerate(searchRay):
        if s == False:
            color[i] = 1

    if distance:
        plt.scatter(rays[:, 0:1], rays[:, 1:2], marker='.', c=hits)
    else:
        plt.scatter(rays[:, 0:1], rays[:, 1:2], marker='.', c=color)

    plt.xlim([-0.35, 0.35])
    plt.ylim([-0.35, 0.35])
    plt.show()
    # """

def sphereRetinaRayCast(rays, pupil, seeLines=False, seeHits=False, seeDistribution=False, nPoints=10000, w=200, h=200):

    nRays = len(rays)

    pcd, scene, line_set = setupScene(seeLines, nPoints, w, h, rays, nRays)

    # create octree
    octree = o3d.geometry.Octree(max_depth=4)                   # > 4 makes search return empty later
    octree.convert_from_point_cloud(pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine

    onv, hits, searchRay = rayCast(rays, nRays, pupil, scene, pcd, octree, seeLines, line_set, seeHits)

    print("# rays that missed: ", np.count_nonzero(searchRay))

    if seeDistribution:
        visualizeHits(rays, hits, searchRay)


#*************************************************************#
# main()
#*************************************************************#

def main():

    retina = np.load('./data/retina_dist.npy')
    retina = retina[:14400]
    retina[:, 2] += 1

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

    problemRays = np.array([
        [0, 0, 2],
        [-0.05305311, 0.03463974,  2.]
    ])

    pupil = np.array([0, 0, 0.5])

    # TODO: delta onv or hits 
    sphereRetinaRayCast(retina, pupil, seeLines=False, seeHits=False, seeDistribution=True)


if __name__=="__main__":
    main()