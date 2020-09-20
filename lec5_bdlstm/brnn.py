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
load = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Model
#MNIST N 1 28 28 
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_class)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _  = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=>Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=>Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint["optimizer"])


#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#init Model
model = BRNN(input_size, hidden_size, num_layers, num_class).to(device)

#Loss ans optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

if load:
  load_checkpoint(torch.load("my_checkpoint.pth.tar"))
#Train
for epoch in range(num_epochs):
    losses = []
    if epoch%1==0:
      checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
      save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        data = data.squeeze(1)
        
        #batch prediction
        pred = model(data)
        loss = criterion(pred, targets)
        if batch_idx%100==0:
          print(f"Epoch: {epoch}/{num_epochs}\tBatch Index:{batch_idx}\tLoss:{loss:0.5f}")
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