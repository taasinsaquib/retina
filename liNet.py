
import torch
from torch.utils import data
import torchvision
import torch.nn as nn
from   torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import random
import pickle
from math import ceil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchSize = 32

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

def generateKNN(retina, nearest, k=2):
    nPhotoreceptors = nearest.shape[0]
    nCenters = ceil(nPhotoreceptors/k)

    # cluster from center outwards 
    dist2center = np.linalg.norm(np.array([0, 0, 0]) - retina, axis=1)
    order = np.argsort(dist2center)

    # make distance matrix
    distances = np.zeros((nPhotoreceptors, nPhotoreceptors))
    for i, r in enumerate(retina):
        distances[i] = np.linalg.norm(r - retina, axis=1)

def generateConnections(nLayers, k = 2):

    # already did some work for the first medians
    nearest = np.load('./data/nearest_3.npy')
    layers = [nearest]
    meds = []

    nPhotoreceptors = 14400
    retina = np.load('./data/retina_dist.npy')[:nPhotoreceptors]

    for i in range(nLayers):
        medians = np.zeros((nearest.shape[0], 3)) # for next round

        used = []

        for i, neighbors in enumerate(nearest):
            neighbors = neighbors.astype(int)
            medians[i] = np.average(retina[neighbors], axis=0)

        meds.append(medians)
        # generate KNN and replace nearest
        # retina becomes medians

        retina = medians
        nPhotoreceptors = retina.shape[0]

        # cluster from center outwards 
        dist2center = np.linalg.norm(np.array([0, 0, 0]) - retina, axis=1)
        order = np.argsort(dist2center)

        nCenters = ceil(nPhotoreceptors/k)
        nearest = np.zeros((nCenters, k))

        # make distance matrix
        distances = np.zeros((nPhotoreceptors, nPhotoreceptors))
        for i, r in enumerate(retina):
            distances[i] = np.linalg.norm(r - retina, axis=1)

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

            count += 1

        layers.append(nearest)

    return layers, meds

def main():
    
    # Run this to get initial neighbors *******************
    # TODO: integrate into the generateConnections() func
    # knn()
    
    # Get connections for deeper layers *******************
    """
    nearest = np.load('./data/nearest_3.npy')
    connections, meds = generateConnections(5)

    with open('./data/connections', 'wb') as f:
      pickle.dump(connections, f)

    with open('./data/meds', 'wb') as f:
      pickle.dump(meds, f)
    """

    # Just used data saved to avoid computing again
    with open('./data/connections', 'rb') as f:
      connections = pickle.load(f)

    with open('./data/meds', 'rb') as f:
      meds = pickle.load(f)

    # Visualize medians ***********************************
    """
    title = 'KNN'
    plt.title(title)
    # plt.scatter(retina[:nPhotoreceptors, 0:1], retina[:nPhotoreceptors, 1:2], marker='.', c=color)
    plt.xlim([-0.35, 0.35])
    plt.ylim([-0.35, 0.35])
    
    for c in connections:
        print(c.shape)

    cols = ['r', 'g', 'b', 'c', 'y']

    for i, m in enumerate(meds):
        print(m.shape)
        plt.scatter(m[:, 0:1], m[:, 1:2], marker='.', c=cols[i])

    plt.show()
    """

    # Prep Data *******************************************
    data_train = np.load(f'./data/data_circle_away_1.0.npy')

    # TODO: turn into degrees or keep radians?
    labels_train = np.load(f'./data/labels_circle_away_1.0.npy') * 180/np.pi
    labels_train = labels_train[:, :2]

    data_train = data_train[:2]
    labels_train = labels_train[:2]
    

    X_train_tensor = torch.from_numpy(data_train).float()
    y_train_tensor = torch.from_numpy(labels_train).float()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    lengths = np.array([0.65, 0.35, 0])
    lengths *= int(len(train_dataset))
    lengths = np.rint(lengths)
    lengths = np.asarray(lengths, dtype=np.int32)

    subset_train, subset_val, _ = random_split(train_dataset, lengths, generator=torch.Generator().manual_seed(28)) 

    train_data = ONVData(
        subset_train, transform=None)

    loader = torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=True,  num_workers=1)
        
    # Create LiNet ****************************************
    m = MyLiNet(connections)
    m.to(torch.float)

    # summary(m, input_size=(32, 14400))

    list(m.parameters())[0].grad

    # inp = np.zeros((32, 14400), dtype=np.float32)
    # inp = torch.tensor(inp)

    # Train ***********************************************

    """
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    for inputs, labels in loader:
        inputs = inputs.float()
        labels = labels.float()

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = m(inputs)
        # print("1", outputs.type())
        # print("2", labels.type())

        # compute loss where the loss function will be defined later
        loss = loss_fn(outputs, labels)

        # print("3", loss.type())

        # backward + optimize only if in training phase
    
        loss.backward()
        optimizer.step()

    out = m.forward(inp)
    print(out)
    """

class ONVData(Dataset):

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):

        x, y = self.subset[index]
        if self.transform:
          x = self.transform(x)

        return x, y

class MyLiNet2(nn.Module):
    def __init__(self, masks):
        super(MyLiNet2, self).__init__()

        self.sparse0 = nn.Linear(14400, 7200)
        self.sparse1 = nn.Linear(7200, 3600)
        self.sparse2 = nn.Linear(3600, 1800)
        self.sparse3 = nn.Linear(1800, 900)
        self.sparse4 = nn.Linear(900, 450)

        self.fc1 = nn.Linear(450, 3)

        self.mask0 = torch.tensor(masks[0])
        self.mask1 = torch.tensor(masks[1])
        self.mask2 = torch.tensor(masks[2])
        self.mask3 = torch.tensor(masks[3])
        self.mask4 = torch.tensor(masks[4])

        # print(self.mask0.shape, self.sparse0.weight.shape)
        # print(self.mask1.shape, self.sparse1.weight.shape)
        # print(self.mask2.shape, self.sparse2.weight.shape)
        # print(self.mask3.shape, self.sparse3.weight.shape)
        # print(self.mask4.shape, self.sparse4.weight.shape)

        # Done by default? try diff initializations
        nn.init.kaiming_normal_(self.sparse0.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.sparse0.bias, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.sparse1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.sparse1.bias, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.sparse2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.sparse2.bias, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.sparse3.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.sparse3.bias, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.sparse4.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.sparse4.bias, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc1.bias, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        self.sparse0.weight.data.mul_(self.mask0)
        x = self.sparse0(x)
        x = torch.relu(x)

        self.sparse1.weight.data.mul_(self.mask1)
        x = self.sparse1(x)
        x = torch.relu(x)

        self.sparse2.weight.data.mul_(self.mask2)
        x = self.sparse2(x)
        x = torch.relu(x)

        self.sparse3.weight.data.mul_(self.mask3)
        x = self.sparse3(x)
        x = torch.relu(x)

        self.sparse4.weight.data.mul_(self.mask4)
        x = self.sparse4(x)
        x = torch.relu(x)

        x = self.fc1(x)

        return x


def kMeans(points, nClusters):

    kmeans = KMeans(n_clusters=nClusters, init='k-means++', n_init=3, random_state=28).fit(points)
    return np.array(kmeans.cluster_centers_), np.array(kmeans.labels_)

def main2():
    nPhotoreceptors = 14400
    retina = np.load('./data/retina_dist.npy')[:nPhotoreceptors, :2]

    # Generate Clusters and Labels
    """
    points = retina
    nClusters = nPhotoreceptors

    for i in range(5):
        nClusters = int(nClusters/2)
        centers, labels = kMeans(points, nClusters)
        
        np.save(f'./data/kmean_centers_{i}', centers)
        np.save(f'./data/kmean_labels_{i}', labels)

        print(centers.shape, labels.shape)

        points = centers
    """

    # Load Clusters and Labels
    c0 = np.load('./data/kmean_centers_0.npy')
    c1 = np.load('./data/kmean_centers_1.npy')
    c2 = np.load('./data/kmean_centers_2.npy')
    c3 = np.load('./data/kmean_centers_3.npy')
    c4 = np.load('./data/kmean_centers_4.npy')

    c0_labels = np.load('./data/kmean_labels_0.npy')
    c1_labels = np.load('./data/kmean_labels_1.npy')
    c2_labels = np.load('./data/kmean_labels_2.npy')
    c3_labels = np.load('./data/kmean_labels_3.npy')
    c4_labels = np.load('./data/kmean_labels_4.npy')

    # Visualize
    # title = 'KNN'
    # plt.title(title)
    # plt.scatter(retina[:, 0:1], retina[:, 1:2], marker='.', c='r')
    # plt.scatter(c0[:, 0:1], c0[:, 1:2], marker='.', c='g')
    # plt.scatter(c1[:, 0:1], c1[:, 1:2], marker='.', c='b')
    # plt.scatter(c2[:, 0:1], c2[:, 1:2], marker='.', c='c')
    # plt.scatter(c3[:, 0:1], c3[:, 1:2], marker='.', c='y')
    # plt.scatter(c4[:, 0:1], c4[:, 1:2], marker='.', c='m')
    # plt.xlim([-0.35, 0.35])
    # plt.ylim([-0.35, 0.35])
    # plt.show()

    """
    c0_labels = np.load('./data/kmean_labels_0.npy')
    nCenters = 7200
    color = [(0, 0, 0)] * nPhotoreceptors
    cs = []
    for i in range(nCenters):
        col = getHexColor()
        cs.append(col)
    
    for i in range(nPhotoreceptors):
        color[i] = cs[c0_labels[i]]

    plt.scatter(retina[:, 0:1], retina[:, 1:2], marker='.', c=color)
    """

    masks = {}

    for i in range(5):
        labels = np.load(f'./data/kmean_labels_{i}.npy')
    
        nPoints = labels.shape[0]
        nClusters = int(nPoints/2)

        mask = np.zeros((nClusters, nPoints), dtype=np.int64)

        for j, l in enumerate(labels):
            mask[l][j] = 1

        masks[i] = mask
        print(i, mask.shape)

        np.save(f'./data/mask_{i}', mask)

    m = MyLiNet2(masks)


      

if __name__ == "__main__":
    # main()

    main2()

# 14400/3 -> 4800/2 -> 2400/2 -> 1200/2 -> 600/2 -> 300/2 -> 150 -> 2

class MyLiNet(nn.Module):
    def __init__(self, connections):
        super(MyLiNet, self).__init__()

        self.connections = connections
        self.linears = {}

        for i, c in enumerate(connections):
            n = c.shape[0]
            k = c.shape[1]

            self.linears[i] = nn.ModuleList([nn.Linear(k, 1) for i in range(n)])

        self.fc1 = nn.Linear(150, 2)

    def forward(self, x):

        prevVals = x

        for i, conn in enumerate(self.connections):

            conn = np.array(conn, dtype=np.int32)

            n = conn.shape[0]
            nBatch = prevVals.size()[0]
            curVals = torch.empty((nBatch, n), device=device)

            for j, c in enumerate(conn):
                
                inp = torch.index_select(prevVals, 1, torch.tensor(c))

                v = self.linears[i][j](inp)
                v = torch.relu(v)

                curVals[:, j*batchSize:(j+1)*batchSize] = v        

            prevVals = curVals

        out = self.fc1(prevVals)

        return out