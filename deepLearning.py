
import torch
import torchvision
import torch.nn as nn

# from torch._C import device # is this needed?

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadModel(path):
    modResnet = torchvision.models.resnet18(pretrained=True)
    modResnet.fc = nn.Linear(512, 3)

    model = MyResNet(modResnet)
    model.load_state_dict(torch.load(path, map_location=device))
    
    model.double()
    model.to(device)
    model.eval()

    return model

class MyResNet(nn.Module):
  def __init__(self, pretrained):
    super(MyResNet, self).__init__()
    
    self.pretrained = pretrained

    self.input = nn.Conv2d(2, 3, (2, 4280), stride=1, padding='same')

  def forward(self, x):

    x = self.input(x)
    x = self.pretrained(x)

    return x

