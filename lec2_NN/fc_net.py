import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets #MNIST etc
import torchvision.transforms as transforms #transform visioin data(rot, trans ...)

class NN(nn.Module):
    def __init__(self, input_size, num_class):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x#log_prob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28*28
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
model = NN(input_size, num_class).to(device)

#Loss ans optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

#Train
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        data = data.reshape(-1, input_size) # reshape
        
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
            x = x.reshape(-1, input_size)
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