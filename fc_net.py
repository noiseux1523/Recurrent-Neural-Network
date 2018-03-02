import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Training settings
nhid = 200
batch_size = 10
nepochs = 10
lr = 0.001

# Load the datasets
train_data = torch.load('data/training.pt')
test_data = torch.load('data/test.pt')

npx = 28 * 28
num_classes = 10

model = nn.Sequential(nn.Linear(npx, nhid), nn.ReLU(),
                      nn.Linear(nhid, nhid), nn.ReLU(),
                      nn.Linear(nhid, num_classes), nn.LogSoftmax())

optimizer = optim.SGD(model.parameters(), lr=lr)


def train(data):
    nsamples = data[0].size(0)
    model.train()
    avg_loss = 0
    perm = torch.randperm(nsamples).long()
    for batch in range(nsamples / batch_size):
        inp = data[0][perm[
                batch * batch_size: (batch + 1) * batch_size]].view(-1, npx).float()
        targ = data[1][perm[
                batch * batch_size: (batch + 1) * batch_size]]
        inp = Variable(inp)
        targ = Variable(targ)
        optimizer.zero_grad()
        output = model(inp)
        loss = F.nll_loss(output, targ)
        avg_loss = (0.9 * avg_loss + 0.1 * loss.data[0]) if batch > 0 else loss.data[0]
        loss.backward()
        optimizer.step()
        if batch % 1000 == 0:
            print('Epoch ', epoch, 'batch', batch, 'loss ', avg_loss)


def test(data):
    model.eval()
    test_loss = 0
    correct = 0
    nsamples = data[0].size(0)
    num_batches = nsamples / batch_size
    for batch in range(num_batches):
        inp = data[0][
            batch * batch_size: (batch + 1) * batch_size, :, :].view(-1, npx).float()
        targ = data[1][
            batch * batch_size: (batch + 1) * batch_size]
        inp, targ = Variable(inp, volatile=True), Variable(targ)
        output = model(inp)
        test_loss += F.nll_loss(output, targ).data[0]
        pred = output.data.max(1)[1] 
        correct += pred.eq(targ.data).sum()
    test_loss /= num_batches  # loss function already averages over batch size
    accuracy = 100. * correct / (num_batches * batch_size)
    print('\nTest Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, num_batches * batch_size, accuracy))

for epoch in range(nepochs):
    train(train_data)
    test(test_data)

