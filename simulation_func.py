import numpy as np

import torch
import torch.nn as nn
import snntorch as snn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nSteps = 1

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.fc1 = nn.Linear(42, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):

        print(x.shape)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        return x

"""
class FCSpiking(nn.Module):
  def __init__(self, alpha, beta):
    super(FCSpiking, self).__init__()

    self.fc1 = nn.Linear(42, 10)
    self.fc2 = nn.Linear(10, 2)

    self.lif1 = snn.Synaptic(alpha=alpha, beta=beta)

  def forward(self, x):

        # Initialize hidden states and outputs at t=0
        syn1, mem1 = self.lif1.init_synaptic()

        # (nSteps, batch, data)
        x = x.permute(1, 0, 2)

        # Record the final layer
        spk1_rec = []
        mem1_rec = []
        self.register_buffer('out_rec', torch.zeros((nSteps, x.size()[1], 2)))
        self.out_rec = torch.zeros((nSteps, x.size()[1], 2)).to(device)

        for step in range(nSteps):
            cur1 = self.fc1(x[step])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

            # print(mem4.size())
            out = self.fc5(mem1)

            # print(self.out_rec.size(), out.size())
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            self.out_rec[step] = out

        return mem1_rec, spk1_rec, self.out_rec
"""

def func(x):
    """
        onv [arrray] 
    """


    onv = np.array(x, dtype=float)
    # print(x)
    # print(type(onv), onv.shape)

    alpha = 0.9
    beta  = 0.8

    m = FC()
    m.to(torch.float)
    m.to(device)

    inputs = torch.from_numpy(onv)
    inputs = inputs.float()

    output = m(inputs)
    output = output.cpu().detach().numpy()

    return output

"""
TODO
    run SNN on onv input
"""