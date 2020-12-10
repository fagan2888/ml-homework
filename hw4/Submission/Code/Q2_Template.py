from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import os
import torch.utils.data as utils

import matplotlib.pyplot as plt

MNIST_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'mnist')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'Figures')

# The parts that you should complete are designated as TODO
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.hidden1 = nn.Linear(28*28, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = F.softmax(self.output(x), dim=1)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    accuracy = test(model, device, train_loader)
    return accuracy

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def main():
    print('Question 2.5')
    print()

    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = False # Switch to False if you only want to use your CPU
    learning_rate = 0.002
    num_epochs = 15
    batch_size = 32

    device = torch.device("cuda" if use_cuda else "cpu")

    train_X = np.load(os.path.join(MNIST_DIR, 'X_train.npy'))
    train_Y = np.load(os.path.join(MNIST_DIR, 'y_train.npy'))

    test_X = np.load(os.path.join(MNIST_DIR, 'X_test.npy'))
    test_Y = np.load(os.path.join(MNIST_DIR, 'y_test.npy'))

#    train_X = train_X.reshape([-1,1,28,28]) # the data is flatten so we reshape it here to get to the original dimensions of images
#    test_X = test_X.reshape([-1,1,28,28])

    # transform to torch tensors
    tensor_x = torch.tensor(train_X, device=device)
    tensor_y = torch.tensor(train_Y, dtype=torch.long, device=device)

    test_tensor_x = torch.tensor(test_X, device=device)
    test_tensor_y = torch.tensor(test_Y, dtype=torch.long)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = utils.DataLoader(test_dataset) # create your dataloader if you get a error when loading test data you can set a batch_size here as well like train_dataloader

    model = NNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accs = []
    test_accs = []
    for epoch in range(num_epochs):
        train_acc = train(model, device, train_loader, optimizer, epoch)
        train_accs.append(train_acc)
        print('\nTrain set Accuracy: {:.0f}%\n'.format(train_acc))
        test_acc = test(model, device, test_loader)
        test_accs.append(test_acc)
        print('\nTest set Accuracy: {:.0f}%\n'.format(test_acc))

    torch.save(model.state_dict(), "mnist_nn.pt")

    # Save test predictions
    # Not sure if/how we're supposed to do this???

    # Plot train and test accuracy vs epoch
    xs = range(num_epochs)
    ys = train_accs
    plt.xlabel('epoch')
    plt.ylabel('training acc')
    plt.plot(xs, ys)
    plt.savefig(os.path.join(FIGURES_DIR, 'training_acc_vs_epoch.png'))
    plt.show()
    
    xs = range(num_epochs)
    ys = test_accs
    plt.xlabel('epoch')
    plt.ylabel('test acc')
    plt.plot(xs, ys)
    plt.savefig(os.path.join(FIGURES_DIR, 'test_acc_vs_epoch.png'))
    plt.show()


if __name__ == '__main__':
    main()
