"""
Author: Honglin Chen
Modification data: August 26, 2018

This script generates the K-nearest neighbor indexes required for locally connection. 
Before running this script, please refer to "Define parameters" section below to specify the paramters (more details will be included in the section)
The output of the KNN indexes will be in ./knn_index
The distributions of the KNN will be shown in ./figure

"""

import numpy as np
import h5py
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

# plt.ion()
import time
import random

# =============================== #
#        Define parameters        #
# =============================== #
"""
1. factor: if the num of neurons in layer L is N, then the num of neurons in layer L+1 will be N/factor
2. k: number of neighbors that are locally connected to each unit
3. num_layer: number of hidden layers, excluding input and output layer
4. R, T: parameters from log-polar mapping
"""

factor = 2
k = 20
num_layer = 5
R = 40
T = 360
dim = R * T
degree_to_radian = np.pi / 180
fov = 240

# Step 1: generate the coordinates for the photoreceptors
# ------------------------------------------------------
maxRadius = 2.6 * np.exp(R / 5.0 * 40.0 / R)
# maxRadius = np.exp(R / 5.0 * 40.0 / R)

r = np.arange(R)
rr = np.repeat(r, T)
t = np.arange(T)
t = t * (360 / T)
tt = np.tile(t, R)

# load the gaussian noise and add noise
noise = np.loadtxt('/Users/Saquib/Desktop/knn/knn/data/denser_retina_distribution_noise.txt')
index = rr + T * tt
noise = np.take(noise, index.tolist())

logPolarX = 2.6 * np.exp(rr / 5.0 * 46.9 / R + noise) * np.cos(degree_to_radian * tt + noise)
logPolarY = 2.6 * np.exp(rr / 5.0 * 46.9 / R + noise) * np.sin(degree_to_radian * tt + noise)

# logPolarX = np.exp(rr / 5.0 * 46.9 / R + noise) * np.cos(degree_to_radian * tt + noise)
# logPolarY = np.exp(rr / 5.0 * 46.9 / R + noise) * np.sin(degree_to_radian * tt + noise)

X = (1 * (logPolarY / maxRadius)) * fov * degree_to_radian
Y = (1 * (logPolarX / maxRadius)) * fov * degree_to_radian

# Step 2: generate the tree for KNN

def construct_ann(coordinate, index):
    t = AnnoyIndex(2, metric='euclidean')  # Length of item vector that will be indexed
    for i in range(coordinate.shape[0]):
      v = coordinate[i, :] 
      t.add_item(i, v)

    t.build(10)  # 10 trees

    with open(f'./tree/tree_{index}.ann', 'w') as f:
      t.save(f'./tree/tree_{index}.ann')

def reduce_indices(coord, num):
    index = np.linspace(0, coord.shape[0], num, endpoint=False)
    index = index.astype(int)
    target = coord[index, :]
    return target

construct_tree = True

coordinate = np.transpose(np.vstack((X, Y)))
print(coordinate.shape)

if construct_tree:
    construct_ann(coordinate, 0)
    input_array = coordinate
    for index in range(num_layer):
        dim = int(dim/factor)
        print(index, input_array.shape, dim)
        target = reduce_indices(input_array, dim)

        print(input_array.shape, target.shape)

        plt.title('My Retina Photoreceptor Distribution')
        plt.scatter(input_array[:, 0], input_array[:, 1], marker='.', color='r')
        plt.scatter(target[:, 0], target[:, 1], marker='.', color='b')
        plt.xlim([-0.35, 0.35])
        plt.ylim([-0.35, 0.35])
        plt.show()

        construct_ann(target, index + 1)
        input_array = target

def getHexColor():
    r = lambda: random.randint(0,255)
    random_number = random.randint(0,16777215)
    hex_number = str(hex(random_number))
    hex_number ='#'+ hex_number[2:]

    return (r()/255., r()/255., r()/255.)

# Compute KNN index

dim = R * T
input_array = coordinate

for index in range(num_layer):
    u = AnnoyIndex(2, metric='euclidean')
    u.load(f'./tree/tree_{index}.ann')
    prev = dim
    dim = int(dim / factor)

    target = reduce_indices(input_array, dim)
    k_nearest = np.zeros((dim, k)).astype(int)

    # ## Visualization
    color = [(0, 0, 0)] * prev
    cs = []

    nCenters = dim
    for i in range(nCenters):
        col = getHexColor()
        cs.append(col)

    # color = ['gray' for mm in range(prev)]
    # size = [0.1 for mm in range(prev)]

    for i in range(dim):
        v = target[i, :]

        r = u.get_nns_by_vector(v, k, search_k=-1, include_distances=True)
        k_nearest[i, :] = np.asarray(r[0])

        # Visualization
        for nn in r[0]:
          color[nn] = cs[i]
        # color[r[0]] = cs[i]
        """
        #color = ['gray' for mm in range(dim * factor)]
        #size = [0.1 for mm in range(dim * factor)]
        self_index = u.get_nns_by_vector(v, 1, search_k=-1, include_distances=True)
        color[self_index[0][0]] = 'blue'
        size[self_index[0][0]] = 20
        L = r[0]
        for nn in L:
            if not nn == self_index[0][0]:
                color[nn] = 'red'
                size[nn] = 20
      """

    # plt.scatter(input_array[:, 0], input_array[:, 1], color=color, marker='.', s=size)
    plt.scatter(input_array[:, 0], input_array[:, 1], color=color, marker='.')
    # plt.savefig('./figure/%d_%d' % (index, i))
    plt.show()
    plt.close()

    input_array = target
    print(k_nearest)

    #np.savetxt('./knn_index_txt/knn_index_%d.txt'%index,k_nearest,delimiter=',')

    with open(f'./tree/knn_index_{index}.h5', 'w') as f:
      h5f = h5py.File(f'./tree/knn_index_{index}.h5', 'w')
      h5f.create_dataset('data', data=k_nearest)
      h5f.close()