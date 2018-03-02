import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ndim = 1
nhid = 200
nout = 1
nsamples = 1000
net = torch.nn.Sequential(nn.Linear(ndim, nhid), nn.ReLU(), nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nout))
print(net)
inputs = torch.arange(-3,3,0.01).view(-1, 1)
outputs = net.forward(Variable(inputs))

fig, ax = plt.subplots()
ax.plot(inputs.squeeze().numpy(), outputs.data.squeeze().numpy())
plt.show()


