import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets #MNIST etc
import torchvision.transforms as transforms #transform visioin data(rot, trans ...)
#HyperParameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256

num_class = 10
lr = 1e-3
batch_size = 64
num_epochs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Model
#MNIST N 1 28 28 
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size = self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # N*time_seq*features
        self.fc = nn.Linear(hidden_size*sequence_length, num_class)
        #instead assume that last hidden state has all informaion of previous hiddens states

        self.fc_short = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)#layer*batchsize*hidden_size

        out, _ = self.gru(x, h0)
        #out = out.reshape(out.shape[0], -1)
        #out = self.fc(out)
        out = self.fc_short(out[:,-1,:])#layer*last_hidden*feature

        return out




#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#init Model
model = GRU(input_size, hidden_size, num_layers, num_class).to(device)

#Loss ans optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

#Train
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        data = data.squeeze(1)
        
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
            x = x.squeeze(1)
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