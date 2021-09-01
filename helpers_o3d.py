import numpy as np
import open3d as o3d

def setupScene(seeLines, nPoints, w, h, rays, nRays):

    # create sphere
    sphereColor = [0.1, 0.1, 0.7]
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color(sphereColor)

    # point cloud (pcd) from mesh (to add to KDTree)
    pcd = mesh_sphere.sample_points_poisson_disk(nPoints)
    # pcd.translate([0, 0, -2.5])
    pcd.translate([0, 0, -5])

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

    # box in front of pupil
    x = 5
    y = 5
    z = -10
    p = [[-x, y, -2], [x, y, -2], [-x, -y, -2], [x, -y, -2],
         [-x, y, z], [x, y, z], [-x, -y, z], [x, -y, z]]
    l = [[0,1], [2,3], [0,2], [1,3],
         [4,5], [6,7], [4,6], [5,8],
         [0, 4], [1, 5], [2, 6], [3, 7]]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(p),
        lines=o3d.utility.Vector2iVector(l),
    )

    scene.add_geometry(ls)

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

    for t in np.arange(0, 22, 0.1):

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