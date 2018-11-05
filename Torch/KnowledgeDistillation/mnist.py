import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from train import train

class Net(nn.Module):
  def __init__(self, n):
    super().__init__()
    self.fc1 = nn.Linear(28*28, n)
    self.fc2 = nn.Linear(n, n)
    self.fc3 = nn.Linear(n, 10)

  def forward(self, x):
    x = x.view(-1, 28*28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def load_data(batch_size=128, num_workers=2):
  T = transforms.Compose([
    transforms.ToTensor(),
  ])
  trainset = datasets.MNIST(root='./data', train=True,
                            download=True, transform=T)
  testset = datasets.MNIST(root='./data', train=False,
                            download=True, transform=T)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers)
  return (trainloader, testloader)

def main():
  model = Net(100)
  trainloader, testloader = load_data()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  loss_fn = nn.CrossEntropyLoss()
  train(model, trainloader, optimizer, loss_fn, testloader,
        epochs=1, save_path="models/mnist.pt")



if __name__ == "__main__":
  main()
