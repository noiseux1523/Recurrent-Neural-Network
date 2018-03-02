import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb

# Training settings
nhid = 10
batch_size = 10
nepochs = 10
lr = 0.005

# Load the datasets
train_data = torch.load('data/training.pt')
test_data = torch.load('data/test.pt')

npx = 28 * 28
num_classes = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, nhid, kernel_size=5)
        self.conv1.bias.data.fill_(0)
        self.conv2 = nn.Conv2d(nhid, nhid * 4, kernel_size=5)
        self.conv2.bias.data.fill_(0)
        self.fc1 = nn.Linear(4*4*(nhid*4), 50)
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(50, 10)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*(nhid*4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=lr)

def train(data):
    nsamples = data[0].size(0)
    model.train()
    avg_loss = 0
    perm = torch.randperm(nsamples).long()
    for batch in range(nsamples / batch_size):
        inp = data[0][perm[
                batch * batch_size: (batch + 1) * batch_size]].float().view(batch_size, 1, 28, 28)
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
            batch * batch_size: (batch + 1) * batch_size, :, :].float().view(batch_size,1, 28, 28)
        targ = data[1][
            batch * batch_size: (batch + 1) * batch_size]
        inp, targ = Variable(inp, volatile=True), Variable(targ)
        output = model(inp)
        test_loss += F.nll_loss(output, targ).data[0]
        pred = output.data.max(1)[1] 
        correct += pred.eq(targ.data).sum()
    test_loss /= num_batches  # loss function already averages over batch size
    accuracy = 100. * correct / (num_batches * batch_size)
    print('\nTest Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, num_batches * batch_size, accuracy))
    return test_loss

old_loss = 0
for epoch in range(nepochs):
    train(train_data)
    new_loss = test(test_data)
    # decrease learning rate if validation error does not decrease
    # (note: in practice, replace test with validation set)
    if epoch > 0:
        if old_loss < new_loss:
            lr = lr / 5
            print("New learning rate is: ", lr)
    old_loss = new_loss

        
