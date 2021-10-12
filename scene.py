import numpy  as np
import open3d as o3d

import time

from eye import Eye

from helpers_general      import vecAngle, generateRandPoint
from helpers_deepLearning import FC1toNN

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
        self.rEye = rEye

        self.nRays = lEye.getNRays()

        # o3d params
        self.nPoints = 10000
        self.w = 200
        self.h = 200

        self.scene    = None
        self.pcd      = None

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

        # scene.get_render_option().light_on = True

        sceneControl = scene.get_view_control()
        sceneControl.set_zoom(1.5)

        # code to visualize rays with a LineSet
        if self.seeLines:
            scene.add_geometry(self.lEye.line_set)
            if self.rEye is not None:
                scene.add_geometry(self.rEye.line_set)

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

    #**************** RUN THE SIM **************#

    def simulate(self):
        # create octree
        octree = o3d.geometry.Octree(max_depth=4)                   # > 4 makes search return empty later

        data   = []
        angles = []
        nZeros = 0

        binaryDeltaOnv = None

        lastCenter = None
        curCenter  = None

        curAngles = None

        # initialize the variables the first time
        octree.convert_from_point_cloud(self.pcd, size_expand=0.01)      # 0.01 is just from the example, seems to work fine

        self.lEye.raycast(octree, self.pcd, self.scene, self.seeLines, self.seeHits)
        if self.rEye is not None:
            self.rEye.raycast(octree, self.pcd, self.scene, self.seeLines, self.seeHits)

        lastCenter = self.pcd.get_center()

        for i in range(4000):

            position1 = generateRandPoint(2.5, 2.5, -10)
            curCenter = self.pcd.get_center()

            # Move AWAY FROM center ****************************************

            # raycast, collect data and label
            translate = position1 - np.array(curCenter)
            # translate = np.array([0, 0.75, 0])
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

            self.lEye.raycast(octree, self.pcd, self.scene, self.seeLines, self.seeHits)
            if self.rEye is not None:
                self.rEye.raycast(octree, self.pcd, self.scene, self.seeLines, self.seeHits)

            curCenter = self.pcd.get_center()

            binaryDeltaOnv = self.lEye.binaryDeltaOnv
            if self.rEye is not None:
                binaryDeltaOnv = np.hstack((binaryDeltaOnv, self.rEye.binaryDeltaOnv))

            rotateSpot = self.lEye.pupil.copy()
            if self.rEye is not None:
                rotateSpot += self.rEye.getPupil()
                rotateSpot = rotateSpot / 2.0
            r, curAngles = vecAngle(rotateSpot, lastCenter, curCenter, polX, polY)
            print(curAngles * 180/np.pi)

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

                # TODO: based only on left eye rn
                if np.count_nonzero(self.lEye.searchRay) > self.nRays - 10:
                    skip = True                
                else:
                    # print(binaryDeltaOnv.shape)
                    data.append(binaryDeltaOnv)
                    angles.append(curAngles[:2])

            if self.seeDistribution:
                self.lEye.visualizeHits(type='events')
                self.lEye.visualizeRGB()
                if self.rEye is not None:
                    self.rEye.visualizeHits(type='events')

            if skip == False:
                self.lEye.moveOnvForward()
                if self.rEye is not None:
                    self.rEye.moveOnvForward()
                lastCenter = curCenter

        self.scene.destroy_window()

        if self.saveData:
            data = np.array(data)
            angles  = np.array(angles)

            np.save(f'data/data_rand',    data)
            np.save(f'data/angles_rand',  angles)

            print(f'#Zero labels: {nZeros}, Data: {data.shape}, Labels: {angles.shape}')


def main():

    pupil = np.array([0., 0., 0.])

    pupilL = np.array([-1, 0, 0])
    pupilR = np.array([ 1, 0, 0])

    # Load retina distribution, shift along z-axis
    retina = np.load('./data/retina_dist.npy')
    retina = retina[:14400]
    retina[:, 2] += 0.5

    # retinaL = retinaR = retina.copy()
    retinaL = np.load('./data/retina_dist.npy')[:14400]
    retinaR = np.load('./data/retina_dist.npy')[:14400]

    retinaL[:, 0:1] -= 1.0
    retinaL[:, 2:3] += 0.5
    retinaR[:, 0:1] += 1.0
    retinaR[:, 2:3] += 0.5

    lEye = Eye(pupilL, retinaL, rgb=True, magnitude=True)
    # rEye = Eye(pupilR, retinaR, rgb=True, magnitude=True)
    rEye = None
    # TODO: make a right eye, figure out how to place them together

    scene = BallScene(lEye, rEye, seeLines=False, seeHits=False, seeDistribution=False, saveData=True)
    scene.setup()

    t1 = time.perf_counter()
    scene.simulate()
    t2 = time.perf_counter()

    print(f'Minutes to complete 1000pts - {(t2-t1) / 60}')

if __name__ == "__main__":
    main()


# TODO
"""
* try new data with LiNet

* lighting, read paper to see 
    all points have the same blue, need normals too?
    randomize lighting spots?

* headless rendering for speed
"""