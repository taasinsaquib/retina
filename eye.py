import numpy  as np
import open3d as o3d
import matplotlib.pyplot as plt

class Eye:
    def __init__(self, pupil, rays, rgb=False, magnitude=False):
        """
            pupil,  [x, y, z], position of pinhole
        """
        self.pupil = pupil
        self.rays  = self.ogRays = rays     # og for visualization because they change when rotating
        self.nRays = len(rays)

        self.rgb       = rgb
        self.magnitude = magnitude

        self.visRetina = np.load('./data/retina_dist.npy')[:14400]

        self.prevOnv = None
        self.curOnv  = None
        self.greyKernel = np.array([0.299, 0.587, 0.114])
        self.binaryDeltaOnv = None                          # misnamed, could be binary or magnitude

        self.hits      = None
        self.searchRay = None

        points = np.zeros((2*self.nRays, 3))
        lines  = [[i, i+1] for i in range(0, self.nRays*2, 2)]
        colors = [[1, 0, 0] for i in range(len(lines))]

        # ray origin points at even indices, odd indices will be replaced with the current end of the ray
        for i in range(0, self.nRays):
            points[i*2] = self.rays[i]

        line_set = o3d.geometry.LineSet(
            points = o3d.utility.Vector3dVector(points),
            lines  = o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        self.line_set = line_set


    #************** GETTERS *********************#

    def getPupil(self):
        return self.pupil

    def getRays(self):
        return self.rays

    def getNRays(self):
        return self.nRays

    def moveOnvForward(self):
        self.prevOnv = self.curOnv

    #************** HELPERS ********************#

    def octreeSearch(self, cur, octree, pcdPoints, seeHits):
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

    def raycast(self, octree, pcd, scene, seeLines, seeHits):

        # setup data structs to hold info
        hits      = [0] * self.nRays                 # hit distances of rays, 0 if no intersection
        searchRay = [True] * self.nRays              # store if corresponding ray index has hit yet
        onv = np.ones((self.nRays, 3))               # store colors of the ray hits, white (1, 1, 1) if not hit 

        for t in np.arange(0, 22, 0.1):

            pcdPoints = np.asanyarray(pcd.points)
            colors = np.asarray(pcd.colors)

            # extend the rays
            curPoints = self.rays * (1-t) + self.pupil * t

            # only search rays who haven't hit yet
            indices = np.argwhere(searchRay)

            # set odd indices of the points array, as mentioned above
            if seeLines:
                for i in indices.flatten():
                    self.line_set.points[2*i+1] = curPoints[i]

            for i in indices.flatten():
                cur = curPoints[i]
                hit, closeIdx = self.octreeSearch(cur, octree, pcdPoints, seeHits)

                if hit == True:
                    onv[i] = colors[closeIdx]

                    hits[i] = t
                    searchRay[i] = False

                    if seeHits:
                        colors[closeIdx, :] = [0, 1, 0]

            # update graphics loop
            scene.update_geometry(pcd)
            if seeLines:
                scene.update_geometry(self.line_set)

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

        if (self.rgb == True):
            onv = onv.flatten('F')
        else:
            onv = np.sum(onv * self.greyKernel, axis=1)

        if self.prevOnv is None:
            self.prevOnv = onv
        else:
            self.curOnv = onv

            if self.magnitude:
                self.convertONVMagnitude()
            else:
                self.convertONVBinary()
            
            # print('r', self.binaryDeltaOnv[0:10])
            # print('g', self.binaryDeltaOnv[14400:14410])
            # print('b', self.binaryDeltaOnv[28800:28810])

        self.hits      = hits
        self.searchRay = searchRay

    # NOTE: prevOnv and curOnv can't be None
    # take the diff in greyscale values, return a vector with {-1, 0, 1} aka events
    def convertONVBinary(self):

        deltaOnv = self.curOnv - self.prevOnv

        negIdx  = np.argwhere(deltaOnv < 0)
        posIdx  = np.argwhere(deltaOnv > 0)
        zeroIdx = np.argwhere(deltaOnv == 0)

        # make deltaOnv into -1, 0, 1
        binaryDeltaOnv = deltaOnv
        binaryDeltaOnv[negIdx]  = -1
        binaryDeltaOnv[posIdx]  = 1
        # binaryDeltaOnv[zeroIdx] = 0

        neg  = len(negIdx)
        pos  = len(posIdx)
        zero = len(zeroIdx)
        print(f'Zero: {zero}, Dim: {neg}, Bright: {pos}')

        self.binaryDeltaOnv = binaryDeltaOnv

        # return binaryDeltaOnv

    def convertONVMagnitude(self):
        # return self.curOnv - self.prevOnv
        self.binaryDeltaOnv = self.curOnv - self.prevOnv

    # plot ray hits and misses, or hit distances
    def visualizeHits(self, type='', distance=True):

        title = ''

        # set default background color for rays that didn't intersect w/ geometry
        color = ['dimgrey'] * len(self.hits)

        if type == 'events' and self.binaryDeltaOnv is not None:
            title = 'Events'

            for i, s in enumerate(self.binaryDeltaOnv):
                if s == -1:
                    # color[i] = 0.5
                    color[i] = 'dodgerblue'
                elif s == 1:
                    # color[i] = 1
                    color[i] = 'coral'

        elif type == 'distance':
            title = 'Distance to Surface'

            # visualize hit distance
            hits = self.hits + np.min(self.hits)
            hits /= np.max(hits)

            color = hits

        else:
            title = 'Hit Locations'
            for i, s in enumerate(self.searchRay):
                if s == False:
                    # color[i] = 1
                    color[i] = 'coral'

        plt.title(title)
        plt.scatter(self.visRetina[:, 0:1], self.visRetina[:, 1:2], marker='.', c=color)
        plt.xlim([-0.35, 0.35])
        plt.ylim([-0.35, 0.35])
        plt.show()

    # repeating code here

    def binary(self, onv):
        negIdx  = np.argwhere(onv < 0)
        posIdx  = np.argwhere(onv > 0)
        zeroIdx = np.argwhere(onv == 0)

        # make deltaOnv into -1, 0, 1
        binaryDeltaOnv = onv
        binaryDeltaOnv[negIdx]  = -1
        binaryDeltaOnv[posIdx]  = 1
        binaryDeltaOnv[zeroIdx] = 0

        neg  = len(negIdx)
        pos  = len(posIdx)
        zero = len(zeroIdx)
        print(f'Zero: {zero}, Dim: {neg}, Bright: {pos}')

        return binaryDeltaOnv

    def color(self, onv):
        color = ['dimgrey'] * len(self.hits)

        for i, s in enumerate(onv):
            if s == -1:
                # color[i] = 0.5
                color[i] = 'dodgerblue'
            elif s == 1:
                # color[i] = 1
                color[i] = 'coral'

        return color
    
    def visualizeRGB(self):
        fig, axs = plt.subplots(1, 3)

        r = self.binaryDeltaOnv[      :14400 ]
        g = self.binaryDeltaOnv[ 14400:28800 ]
        b = self.binaryDeltaOnv[ 28800:      ]

        print("red")
        rBin = self.binary(r)
        print("green")
        gBin = self.binary(g)
        print("blue")
        bBin = self.binary(b)

        rCol = self.color(rBin)
        gCol = self.color(gBin)
        bCol = self.color(bBin)
        
        axs[0].set_title('Red?')
        axs[0].set_xlim([-0.35, 0.35])
        axs[0].set_ylim([-0.35, 0.35])
        axs[0].scatter(self.visRetina[:, 0:1], self.visRetina[:, 1:2], marker='.', c=rCol)

        axs[1].set_title('Green?')
        axs[1].set_xlim([-0.35, 0.35])
        axs[1].set_ylim([-0.35, 0.35])
        axs[1].scatter(self.visRetina[:, 0:1], self.visRetina[:, 1:2], marker='.', c=gCol)

        axs[2].set_title('Blue?')
        axs[2].set_xlim([-0.35, 0.35])
        axs[2].set_ylim([-0.35, 0.35])
        axs[2].scatter(self.visRetina[:, 0:1], self.visRetina[:, 1:2], marker='.', c=bCol)

        plt.show()


def main():
    pass

if __name__ == "__main__":
    main()
