
import torch
import torchvision
import torch.nn as nn

import numpy as np

# from torch._C import device # is this needed?

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyResNet(nn.Module):
  def __init__(self, pretrained):
    super(MyResNet, self).__init__()
    
    self.pretrained = pretrained

    self.input = nn.Conv2d(2, 3, (2, 4280), stride=1, padding='same')

  def forward(self, x):

    x = self.input(x)
    x = self.pretrained(x)

    return x

def loadModel(path):
    modResnet = torchvision.models.resnet18(pretrained=True)
    modResnet.fc = nn.Linear(512, 3)

    model = MyResNet(modResnet)
    model.load_state_dict(torch.load(path, map_location=device))
    
    model.double()
    model.to(device)
    model.eval()

    return model

# run NN with ONV input
def onvToNN(model, binaryDeltaOnv):
    squareOnv = np.reshape(binaryDeltaOnv, (120, 120))  # 14400 photoreceptors reshaped to a 120*120 square
    channelEvents = np.zeros((1, 2, 120, 120))

    bright = np.argwhere(squareOnv > 0)
    dim    = np.argwhere(squareOnv < 0)

    channelEvents[0][0][bright[:, :2]] = 1
    channelEvents[0][1][dim[:, :2]]    = -1

    input = torch.from_numpy(channelEvents)
    with torch.no_grad():
        output = model(input)

    return output.cpu().numpy()

#******************************#
# Data Collection / Processing
#******************************#

# take the diff in greyscale values, reutrn a vector with {-1, 0, 1} aka events
def convertONV(curOnv, prevOnv):

    deltaOnv = curOnv - prevOnv

    negIdx  = np.argwhere(deltaOnv < 0)
    posIdx  = np.argwhere(deltaOnv > 0)
    zeroIdx = np.argwhere(deltaOnv == 0)

    # make deltaOnv into -1, 0, 1
    binaryDeltaOnv = deltaOnv
    binaryDeltaOnv[negIdx]  = -1
    binaryDeltaOnv[posIdx]  = 1
    binaryDeltaOnv[zeroIdx] = 0

    neg  = len(negIdx)
    pos  = len(posIdx)
    zero = len(zeroIdx)
    print(f'Zero: {zero}, Dim: {neg}, Bright: {pos}')

    return binaryDeltaOnv