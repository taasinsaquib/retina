import numpy as np
import open3d as o3d

def f_traverse(node, node_info):
    return True
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop

w = 200
h = 200

# create sphere
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.compute_vertex_normals()
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

# Point Cloud from sphere mesh
N = 20000
pcd = mesh_sphere.sample_points_uniformly(N)
# o3d.visualization.draw_geometries([pcd])

# create scene, add sphere to it
scene = o3d.visualization.VisualizerWithKeyCallback()
scene.create_window(window_name='Main Scene', width=w, height=h, left=200, top=500, visible=True)
scene.add_geometry(mesh_sphere)

sceneControl = scene.get_view_control()
sceneControl.set_zoom(1.5)

# oct scene
octScene = o3d.visualization.Visualizer()
octScene.create_window(window_name='See Movement', width=w, height=h, left=200, top=700, visible=True)
octScene.add_geometry(pcd)

octControl = octScene.get_view_control()
octControl.set_zoom(1.5)

# figure out good depth
octree = o3d.geometry.Octree(max_depth=1)

# "rays"
origins = [[0, 0, -0.5]]
directions = [[0, 0, 1]]

while True:
    mesh_sphere.translate((0.01, 0, 0))
    pcd.translate((0.01, 0, 0))
    # octree.translate((0.1, 0, 0))    # "not implemented"

    # hidden point removal for pcd
    # diameter = np.linalg.norm( np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()) )
    # print(diameter)
    # camera = [0, 0, diameter]
    # radius = diameter * 100
    # _, pt_map = pcd.hidden_point_removal(camera, radius)
    # pcd = pcd.select_by_index(pt_map)

    scene.update_geometry(mesh_sphere)
    scene.poll_events()
    scene.update_renderer()

    octScene.update_geometry(pcd)
    octScene.poll_events()
    octScene.update_renderer()

    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    # o3d.visualization.draw_geometries([octree])
    octree.traverse(f_traverse)

    octree.clear()

scene.destroy_window()
octScene.destroy_window()
   