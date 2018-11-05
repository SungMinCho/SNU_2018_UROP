import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from train import train

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

class Teacher(nn.Sequential):
  def __init__(self):
    super().__init__()
    # input shape , (B, 32, 32, 3)
    self.add_module("conv1_1" , nn.Conv2d(3, 64, 3, padding=1))
    self.add_module("bn1_1" , nn.BatchNorm2d(64))
    self.add_module("dr1_1" , nn.Dropout2d(0.3))
    self.add_module("conv1_2" , nn.Conv2d(64, 64, 3, padding=1))
    self.add_module("bn1_2" , nn.BatchNorm2d(64))
    self.add_module("pool1_1" , nn.MaxPool2d(2, 2))
    # (B, 16, 16, 64)
    self.add_module("conv2_1" , nn.Conv2d(64, 128, 3, padding=1))
    self.add_module("bn2_1" , nn.BatchNorm2d(128))
    self.add_module("dr2_1" , nn.Dropout2d(0.4))
    self.add_module("conv2_2" , nn.Conv2d(128, 128, 3, padding=1))
    self.add_module("bn2_2" , nn.BatchNorm2d(128))
    self.add_module("pool2_1" , nn.MaxPool2d(2, 2))
    # (B, 8, 8, 128)
    self.add_module("conv3_1" , nn.Conv2d(128, 256, 3, padding=1))
    self.add_module("bn3_1" , nn.BatchNorm2d(256))
    self.add_module("dr3_1" , nn.Dropout2d(0.4))
    self.add_module("conv3_2" , nn.Conv2d(256, 256, 3, padding=1))
    self.add_module("bn3_2" , nn.BatchNorm2d(256))
    self.add_module("dr3_2" , nn.Dropout2d(0.4))
    self.add_module("conv3_3" , nn.Conv2d(256, 256, 3, padding=1))
    self.add_module("bn3_3" , nn.BatchNorm2d(256))
    self.add_module("pool3_1" , nn.MaxPool2d(2, 2))
    # (B, 4, 4, 256)
    self.add_module("conv4_1" , nn.Conv2d(256, 512, 3, padding=1))
    self.add_module("bn4_1" , nn.BatchNorm2d(512))
    self.add_module("dr4_1" , nn.Dropout2d(0.4))
    self.add_module("conv4_2" , nn.Conv2d(512, 512, 3, padding=1))
    self.add_module("bn4_2" , nn.BatchNorm2d(512))
    self.add_module("dr4_2" , nn.Dropout2d(0.4))
    self.add_module("conv4_3" , nn.Conv2d(512, 512, 3, padding=1))
    self.add_module("bn4_3" , nn.BatchNorm2d(512))
    self.add_module("pool4_1" , nn.MaxPool2d(2, 2))
    # (B, 2, 2, 512)
    self.add_module("conv5_1" , nn.Conv2d(512, 512, 3, padding=1))
    self.add_module("bn5_1" , nn.BatchNorm2d(512))
    self.add_module("dr5_1" , nn.Dropout2d(0.4))
    self.add_module("conv5_2" , nn.Conv2d(512, 512, 3, padding=1))
    self.add_module("bn5_2" , nn.BatchNorm2d(512))
    self.add_module("dr5_2" , nn.Dropout2d(0.4))
    self.add_module("conv5_3" , nn.Conv2d(512, 512, 3, padding=1))
    self.add_module("bn5_3" , nn.BatchNorm2d(512))
    self.add_module("pool5_1" , nn.MaxPool2d(2, 2))
    self.add_module("dr5_3" , nn.Dropout2d(0.5))
    # (B, 1, 1, 512)
    # flatten here
    self.add_module("flatten6_1", Flatten())
    self.add_module("fc6_1" , nn.Linear(512, 512))
    self.add_module("bn6_1" , nn.BatchNorm1d(512))
    self.add_module("dr6_1" , nn.Dropout(0.5))
    self.add_module("fc2" , nn.Linear(512, 10))

def load_data(batch_size=128, num_workers=2):
  T = transforms.Compose([
    transforms.ToTensor(),
  ])

  trainset = datasets.CIFAR10(root='./data', train=True,
                              download=True, transform=T)
  trainset = datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=T)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
  testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
  return (trainloader, testloader)

def main():
  teacher = Teacher()
  trainloader, testloader = load_data()
  optimizer = optim.Adam(teacher.parameters(), lr=0.001)
  loss_fn = nn.CrossEntropyLoss()
  train(teacher, trainloader, optimizer, loss_fn, testloader,
        epochs=1, save_path="models/cifar10teacher.pt")

if __name__ == "__main__":
  main()
