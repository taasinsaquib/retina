import torch.nn as nn

class MyResNet(nn.Module):
  def __init__(self, pretrained):
    super(MyResNet, self).__init__()
    
    self.pretrained = pretrained

    self.input = nn.Conv2d(2, 3, (2, 4280), stride=1, padding='same')

  def forward(self, x):

    x = self.input(x)
    x = self.pretrained(x)

    return x