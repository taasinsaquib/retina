
# import torch
# import torchvision
# import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import random

from math import ceil

def getHexColor():
    r = lambda: random.randint(0,255)
    random_number = random.randint(0,16777215)
    hex_number = str(hex(random_number))
    hex_number ='#'+ hex_number[2:]

    return (r()/255., r()/255., r()/255.)


def knn():

    nPhotoreceptors = 14400
    retina = np.load('./data/retina_dist.npy')[:nPhotoreceptors]
    color = [(0, 0, 0)] * nPhotoreceptors
    cs = []

    k = 3

    nCenters = ceil(nPhotoreceptors/k)
    for i in range(nCenters):
        col = getHexColor()
        cs.append(col)

    dist2center = np.linalg.norm(np.array([0, 0, 0]) - retina, axis=1)
    order = np.argsort(dist2center)

    """
    # make distance matrix
    distances = np.zeros((nPhotoreceptors, nPhotoreceptors))
    for i, r in enumerate(retina):
        distances[i] = np.linalg.norm(r - retina, axis=1)
    # np.save('./data/distances', distances)
    # distances = np.load('./data/distances.npy')

    nearest = np.zeros((nCenters, k))

    used = []
    count = 0

    # go from the center out
    for i in order:
        if i in used:
            continue

        r = distances[i]
        
        neighbors = np.argpartition(r, k)[:k]
        used += neighbors.tolist()
        nearest[count] = neighbors
        distances[:, neighbors] = 100   # infinity

        for j in neighbors.astype(int):
            color[j] = cs[count]
        count += 1

    np.save('./data/nearest_3', nearest)
    """

    nearest = np.load('./data/nearest_3.npy')

    medians = np.zeros((nearest.shape[0], 3)) # for next round

    used = []
    count = 0

    for neighbors in nearest:
        neighbors = neighbors.astype(int)
        medians[count] = np.average(retina[neighbors], axis=0)
        for j in neighbors:
            color[j] = cs[count]
        count += 1

    title = 'KNN'
    plt.title(title)
    plt.scatter(retina[:nPhotoreceptors, 0:1], retina[:nPhotoreceptors, 1:2], marker='.', c=color)
    # plt.scatter(medians[:, 0:1], medians[:, 1:2], marker='.', c='red')
    plt.xlim([-0.35, 0.35])
    plt.ylim([-0.35, 0.35])
    plt.show()


def main():
    knn()


if __name__ == "__main__":
    main()

# TODO: generate indices for the other layers
#  https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList
# 14400/3 -> 4800/2 -> 2400/2 -> 1200/2 -> 600/2 -> 300 -> 2