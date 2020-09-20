import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets #MNIST etc
import torchvision.transforms as transforms #transform visioin data(rot, trans ...)

torch.manual_seed(0)

class CNN(nn.Module):
    def __init__(self, in_channel=1, num_class=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channel, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #sample convolution
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))# 14*14
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #sample convolution
        self.fc1 = nn.Linear(16*7*7, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x) 

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x#log_prob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_class = 10
lr = 1e-3
batch_size = 64
num_epochs = 1

#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#init Model
model = CNN().to(device)

#Loss ans optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

#Train
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        #batch prediction
        pred = model(data)
        loss = criterion(pred, targets)

        #back prop
        optimizer.zero_grad()
        loss.backward()

        optimizer.step() # update weight thetha <- thetha + lr * grad or other fancy method


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on train data")
    else:
        print("checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval() #freeze network

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")


    model.train()
    acc = float(num_correct)/float(num_samples)
    return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)