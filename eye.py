import numpy  as np
import open3d as o3d
import matplotlib.pyplot as plt

from helpers_general      import vecAngle, generateRandPoint
from helpers_deepLearning import convertONV, FC1toNN

class Eye:
    def __init__(self, pupil, rays):
        """
            pupil,  [x, y, z], position of pinhole
        """
        self.pupil = pupil
        self.rays  = rays
        self.nRays = len(rays)

    #************** GETTERS *********************#

    def getPupil(self):
        return self.pupil

    def getRays(self):
        return self.rays

    def getNRays(self):
        return self.nRays

    #************** HELPERS ********************#
    # plot ray hits and misses, or hit distances
    def visualizeHits(self, rays, hits, searchRay, binaryHits, type='', distance=True):

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

    def cast():
        pass

class BallScene:
    def __init__(self, lEye, rEye=None, seeLines=False, seeHits=False, seeDistribution=False, saveData=False, model=None):
        """
            seeLines,        bool
            seeHits,         bool
            seeDistribution, bool
            saveData,        bool
        """
        self.seeLines        = seeLines
        self.seeHits         = seeHits
        self.seeDistribution = seeDistribution
        self.saveData        = saveData
        
        self.model = model

        self.lEye = lEye
        # self.rEye = rEye

        self.nRays = lEye.getNRays()

        self.greyKernel = np.array([0.299, 0.587, 0.114])

        # o3d params
        self.nPoints = 10000
        self.w = 200
        self.h = 200

        self.scene    = None
        self.pcd      = None
        self.line_set = None

    #************** GETTERS *********************#

    def setSeeLines(self, seeLines=True):
        self.seeLines = seeLines

    def setSeeHits(self, seeHits=True):
        self.seeHits = seeHits

    def setSeeDistribution(self, seeDistribution=True):
        self.seeDistribution = seeDistribution

    def setSaveData(self, saveData=True):
        self.saveData = saveData

    #************** O3D HELPERS ****************#

    def setup(self):
        # create sphere
        sphereColor = [0.1, 0.1, 0.7]
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(sphereColor)

        # point cloud (pcd) from mesh (to add to KDTree)
        pcd = mesh_sphere.sample_points_poisson_disk(self.nRays)
        # pcd.translate([0, 0, -2.5])
        pcd.translate([0, 0, -5])

        # set up scene with pcd
        scene = o3d.visualization.VisualizerWithKeyCallback()
        scene.create_window(window_name='Main Scene', width=self.w, height=self.h, left=200, top=100, visible=True)
        scene.add_geometry(pcd)

        sceneControl = scene.get_view_control()
        sceneControl.set_zoom(1.5)

        line_set = None

        # code to visualize rays with a LineSet
        if self.seeLines:
            points = np.zeros((2*self.nRays, 3))
            lines = [[i, i+1] for i in range(0, self.nRays*2, 2)]
            colors = [[1, 0, 0] for i in range(len(lines))]

            # ray origin points at even indices, odd indices will be replaced with the current end of the ray
            for i in range(0, self.nRays):
                points[i*2] = self.lEye.getRays()[i]

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

        self.scene    = scene
        self.pcd      = pcd
        self.line_set = line_set

    def octreeSearch(self, cur, octree, pcdPoints):
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
                if self.seeHits:
                    closeIdx = np.argmin(dist)
                    idx = leaf.indices[closeIdx]

        return found, idx

    def raycast(self, octree):
        # setup data structs to hold info
        hits      = [0] * self.nRays                 # hit distances of rays, 0 if no intersection
        searchRay = [True] * self.nRays              # store if corresponding ray index has hit yet
        onv = np.ones((self.nRays, 3))               # store colors of the ray hits, white (1, 1, 1) if not hit 

        for t in np.arange(0, 22, 0.1):

            pcdPoints = np.asanyarray(self.pcd.points)
            colors = np.asarray(self.pcd.colors)

            # extend the rays
            curPoints = self.lEye.rays * (1-t) + self.lEye.pupil * t

            # only search rays who haven't hit yet
            indices = np.argwhere(searchRay)

            # set odd indices of the points array, as mentioned above
            if self.seeLines:
                for i in indices.flatten():
                    self.line_set.points[2*i+1] = curPoints[i]

            for i in indices.flatten():
                cur = curPoints[i]
                hit, closeIdx = self.octreeSearch(cur, octree, pcdPoints)

                if hit == True:
                    onv[i] = colors[closeIdx]
                    
                    hits[i] = t
                    searchRay[i] = False

                    if self.seeHits:
                        colors[closeIdx, :] = [0, 1, 0]

            # update graphics loop
            self.scene.update_geometry(self.pcd)
            if self.seeLines:
                self.scene.update_geometry(self.line_set)

            self.scene.poll_events()
            self.scene.update_renderer()

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

    #**************** RUN THE SIM **************#

    def simulate(self):
        # create octree
        octree = o3d.geometry.Octree(max_depth=4)                   # > 4 makes search return empty later

        data   = []
        angles = []
        nZeros = 0

        lastOnv = None
        curOnv  = None
        binaryDeltaOnv = None

        lastCenter = None
        curCenter  = None

        curAngles = None
        ogRays = self.lEye.getRays()       # points get rotated around

        # initialize the variables the first time
        octree.convert_from_point_cloud(self.pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine

        onv, hits, searchRay = self.raycast(octree)
        lastOnv = np.sum(onv * self.greyKernel, axis=1)
        lastCenter = self.pcd.get_center()

        for i in range(1000):

            position1 = generateRandPoint(2.5, 2.5, -10)
            curCenter = self.pcd.get_center()

            # Move AWAY FROM center ****************************************

            # raycast, collect data and label
            translate = position1 - np.array(curCenter)
            # print(position1, curCenter, translate)
            self.pcd.translate(translate)

            # system to move the sphere across the screen
            x = translate[0]
            y = translate[1]

            polX = polY = 1
            if x < 0:
                polX = -1

            if y < 0:
                polY = -1

            octree.clear()
            octree.convert_from_point_cloud(self.pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine
            
            # TODO: put this functionality inside Eye
            
            onv, hits, searchRay = self.raycast(octree)
            curOnv = np.sum(onv * self.greyKernel, axis=1)
            curCenter = self.pcd.get_center()
            
            # TODO TODO TODO
            
            binaryDeltaOnv = convertONV(curOnv, lastOnv)
            # binaryDeltaOnv = convertONVDiff(curOnv, lastOnv)

            r, curAngles = vecAngle(self.lEye.getPupil(), lastCenter, curCenter, polX, polY)

            skip = False

            # inference with nn
            if self.model is not None:

                output = FC1toNN(self.model, binaryDeltaOnv)

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
            elif self.saveData:

                if np.count_nonzero(searchRay) > self.nRays - 10:
                    skip = True                
                else:
                    data.append(binaryDeltaOnv)
                    angles.append(curAngles[:2])

            if self.seeDistribution:
                self.lEye.visualizeHits(ogRays, hits, searchRay, binaryDeltaOnv, type='events')

            if skip == False:
                lastOnv = curOnv
                lastCenter = curCenter

        self.scene.destroy_window()

        if self.saveData:
            data = np.array(data)
            angles  = np.array(angles)

            np.save(f'data/data_rand',    data)
            np.save(f'data/angles_rand',  angles)

            print(f'#Zero labels: {nZeros}, Data: {data.shape}, Labels: {angles.shape}')


def main():
    pupil = np.array([0, 0, 0])

    # Load retina distribution, shift along z-axis
    retina = np.load('./data/retina_dist.npy')
    retina = retina[:14400]
    retina[:, 2] += 0.5

    lEye = Eye(pupil, retina)

    scene = BallScene(lEye, None, seeLines=True, seeHits=False, seeDistribution=True, saveData=False)
    scene.setup()
    scene.simulate()

if __name__ == "__main__":
    main()