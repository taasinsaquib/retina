import numpy as np
from scipy.spatial.transform import Rotation as R

import random

random.seed(528)

def axisAngle(v1, v2):

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return angle

# https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def vecAngle(pupil, lastCenter, curCenter, polX, polY):

    v1 = lastCenter - pupil
    v2 = curCenter - pupil

    x = axisAngle([v1[1], v1[2]], [v2[1], v2[2]])
    y = axisAngle([v1[0], v1[2]], [v2[0], v2[2]])
    z = axisAngle([v1[0], v1[1]], [v2[0], v2[1]])

    angles = np.array([polY * x, polX * -y, 0])
    r = R.from_euler('xyz', angles)

    # return the rotation matrix
    return np.array(r.as_matrix()), angles


def makeCircleXY(r, sampleRate=0.01):
    # x^2 + y^2 = r^2

    x = np.arange(-r, r+sampleRate, sampleRate) # add sampleRate to get the last point
    
    ySquared = (r**2 - np.square(x))
    ySquared[ySquared < 1e-4] = 0       # prevent negative numbers
    y = np.sqrt(ySquared)               # plus/minus

    x = np.hstack((x, x))
    y = np.hstack((y, -1*y))

    return zip(x, y)


def generateRandPoint(x, y, z):

    # thousandth place
    xCoord = round(random.uniform(-x,x), 3)
    yCoord = round(random.uniform(-y,y), 3)
    zCoord = round(random.uniform(z,-2), 3)

    return np.array([xCoord, yCoord, zCoord])
