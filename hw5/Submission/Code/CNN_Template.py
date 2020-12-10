from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torchsummary import summary # todo: remove

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as utils
import pandas as pd

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*12*12, out_features=128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        #x = F.softmax(self.fc2(x), dim=1) # softmax isn't necessary?
        x = self.fc2(x)
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
    accuracy, _ = test(model, device, train_loader)
    return accuracy

def test(model, device, test_loader):
    model.eval()
    correct = 0
    preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            preds.extend([p[0] for p in pred])

    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy, preds


def main():
    torch.manual_seed(1)
    np.random.seed(1)

    # Training settings
    use_cuda = False # Switch to False if you only want to use your CPU
    learning_rate = 0.01
    NumEpochs = 10
    batch_size = 32

    device = torch.device("cuda" if use_cuda else "cpu")

    train_X = np.load('../../Data/X_train.npy')
    train_Y = np.load('../../Data/y_train.npy')

    test_X = np.load('../../Data/X_test.npy')
    test_Y = np.load('../../Data/y_test.npy')

    train_X = train_X.reshape([-1,1,28,28]) # the data is flatten so we reshape it here to get to the original dimensions of images
    test_X = test_X.reshape([-1,1,28,28])

    # transform to torch tensors
    tensor_x = torch.tensor(train_X, device=device)
    tensor_y = torch.tensor(train_Y, dtype=torch.long, device=device)

    test_tensor_x = torch.tensor(test_X, device=device)
    test_tensor_y = torch.tensor(test_Y, dtype=torch.long)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = utils.DataLoader(test_dataset) # create your dataloader if you get a error when loading test data you can set a batch_size here as well like train_dataloader

    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #print(summary(model, input_size=(1, 28, 28))) # todo: remove this

    train_accs = []
    test_accs = []

    for epoch in range(NumEpochs):
        train_acc = train(model, device, train_loader, optimizer, epoch)
        print('\nTrain set Accuracy: {:.1f}%\n'.format(train_acc))
        test_acc, test_preds = test(model, device, test_loader)
        print('\nTest set Accuracy: {:.1f}%\n'.format(test_acc))
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
    torch.save(model.state_dict(), "mnist_cnn.pt")

    # save the final test preds
    pd.Series(test_preds).to_csv('../Predictions/best.csv', index=False)

    xs, ys = range(NumEpochs), train_accs
    plt.xlabel('epoch')
    plt.ylabel('training accuracy')
    plt.plot(xs, ys)
    plt.savefig('../Figures/train_acc_vs_epoch.png')
    plt.show()
    
    xs, ys = range(NumEpochs), test_accs
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.plot(xs, ys)
    plt.savefig('../Figures/test_acc_vs_epoch.png')
    plt.show()


if __name__ == '__main__':
    main()
