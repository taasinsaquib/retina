import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def sphereRetinaRayCast(rays, pupil, seeLines=True, nPoints=10000, w=200, h=200):
    
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

    # data structs to hold info
    nRays = len(rays)
    hits =  [0] * nRays
    searchRay = [True] * nRays
    onv = np.ones((nRays, 3))

    # try octtree
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    print(octree.get_max_bound())
    print(octree.get_min_bound())

    print(octree.size)
    print(octree.origin)

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

    for t in np.arange(0, 10, 0.1):
        # not sure if this updates the tree (probs not?) figure out how to move, probs another nested for outside of this one
        # pcd.translate([0.1, 0, 0])
        pcdPoints = np.asanyarray(pcd.points)

        # Add current point cloud to Tree (not sure if we can just translate the pcd in the tree or something to avoid re-creating the tree in case the sphere moves)
        # creating it with 10k points is slow per frame
        tree = o3d.geometry.KDTreeFlann()
        tree.set_geometry(pcd)

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

            # can't traverse tree directly so I'm using this method
            k, idx, _ = tree.search_hybrid_vector_3d(cur, 0.05, 1)

            if k > 0:
                # print("hit ", t, k, idx, colors[idx[1:], :])

                curColor = colors[idx[1:], :]

                # because of poisson sample, rays go through gaps?
                if len(curColor) == 0:
                    onv[i] = sphereColor
                else:
                    onv[i] = curColor
                
                hits[i] = t
                searchRay[i] = False

                # if seeLines:
                    # colors[idx[:1], :] = [1, 0, 0]

            # octtree
            leaf, info = octree.locate_leaf_node(cur)
            # print(leaf, info)
            if leaf is not None:
                # print("HIT", octree.is_point_in_bound(cur, octree.origin, octree.size), cur, leaf.color, leaf.indices)
                print("HIT", i, t)
                candidates = pcdPoints[leaf.indices]

                dist = np.linalg.norm(candidates - cur, axis=1)

                if np.count_nonzero(dist < 0.1) > 0:

                    hits[i] = t
                    searchRay[i] = False

                    closeIdx = np.argmin(dist)
                    # print(cur)
                    # print(candidates)
                    # print(dist)
                    print(closeIdx)
                
                    colors[leaf.indices[closeIdx], :] = [0, 1, 0]
                    

        # update graphics loop
        scene.update_geometry(pcd)
        if seeLines:
            scene.update_geometry(line_set)

        scene.poll_events()
        scene.update_renderer()
        # scene.run()

    scene.run()
    scene.destroy_window()

    return onv, hits, searchRay

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

    onv, hits, searchRay = sphereRetinaRayCast(retina, pupil, False)

    # see hits on retina distribution

    print("See HITS")

    print("# rays that missed: ", np.count_nonzero(searchRay))
    """
    # visualize hit distance
    hits += np.min(hits)
    hits /= np.max(hits)

    # just visualize if there was a hit or not
    color = np.zeros(len(hits))
    for i, s in enumerate(searchRay):
        if s == False:
            color[i] = 1

    plt.scatter(retina[:, 0:1], retina[:, 1:2], marker='.', c=color)
    # plt.scatter(retina[:, 0:1], retina[:, 1:2], marker='.', c=hits)
    plt.xlim([-0.35, 0.35])
    plt.ylim([-0.35, 0.35])
    plt.show()
    """

if __name__=="__main__":
    main()