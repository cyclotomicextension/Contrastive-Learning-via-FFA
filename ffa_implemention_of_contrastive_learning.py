# -*- coding: utf-8 -*-
"""Forward Forward ALGORITHM
-Implemention of Contrastive Learning on MNIST Dataset

Originally created on colaboratory

Author: Additi Pandey
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 100
batch_size = 100
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


class RepresentationLearningCNN(nn.Module):
    def __init__(self):
        super(RepresentationLearningCNN, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer to flatten the output from previous layer to 1D vector
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)

        return x

# Instantiate the model
model = RepresentationLearningCNN()

class LTrans(nn.Module):
  def __init__(self, num_classes):
    super(LTrans, self).__init__()
    self.linear_layer = nn.Linear(in_features= 57600, out_features=10)

  def forward(self, representation_vectors):
    # Use the linear layer to convert the representation vectors into logits
    logits = self.linear_layer(representation_vectors)
    return logits

linear_trans=LTrans(10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create the filter
filter = torch.tensor([[1/4, 1/2, 1/4]])

# Repeat the filter along the channel axis
filter = filter.unsqueeze(0).unsqueeze(0)

# Move the filter tensor to the GPU
filter = filter.to(device)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
  for (images, labels) in (train_loader):
# origin shape: [100, 1, 28, 28]
# resized: [100, 784]
    images = images.reshape(-1, 1, 28, 28).to(device)
    labels = labels.to(device)
    images = F.conv2d(images, filter, padding=1)
    images = F.conv2d(images, filter.transpose(2, 3), padding=1)
    mask = (images > 0.5).float()
    neg_data = (images * mask) + (images * mask.transpose(-2,-1))
    pos_data = images
    combined_data = torch.cat((pos_data, neg_data), dim=0).to(device)
    combined_data = combined_data[:100]

    representation_vectors = model.conv1(combined_data).relu().to(device)
    representation_vectors = model.conv2(representation_vectors).relu().to(device)
    representation_vectors = representation_vectors.view(representation_vectors.shape[0], -1)
    representation_vectors = model.flatten(representation_vectors).to(device)
    outputs = linear_trans(representation_vectors).to(device)
    loss = criterion(outputs, labels)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in train_loader:
    labels = labels.to(device)
    images = images.reshape(-1, 1, 28, 28).to(device)
    images = F.conv2d(images, filter, padding=1).to(device)
    images = F.conv2d(images, filter.transpose(2, 3), padding=1).to(device)
    mask = (images > 0.5).float().to(device)
    neg_data = (images * mask) + (images * mask.transpose(-2,-1)).to(device)
    pos_data = images.to(device)
    combined_data = torch.cat((pos_data, neg_data), dim=0).to(device)
    indices = torch.randperm(combined_data.size(0))  
    combined_data_shuffled = combined_data[indices]  
    combined_data = combined_data[:100].to(device)
    representation_vectors = model.conv1(combined_data).relu().to(device)
    representation_vectors = model.conv2(representation_vectors).relu().to(device)
    representation_vectors = representation_vectors.view(representation_vectors.shape[0], -1).to(device)
    representation_vectors = model.flatten(representation_vectors).to(device)
    outputs = linear_trans(representation_vectors).to(device)
    _, predicted = torch.max((outputs.data), 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
  print('Accuracy of the network on the training images: {} %'.format( 100 * correct / total))

with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    labels = labels.to(device)
    images = images.reshape(-1, 1, 28, 28).to(device)
    images = F.conv2d(images, filter, padding=1).to(device)
    images = F.conv2d(images, filter.transpose(2, 3), padding=1).to(device)
    mask = (images > 0.5).float().to(device)
    neg_data = (images * mask) + (images * mask.transpose(-2,-1)).to(device)
    pos_data = images.to(device)
    combined_data = torch.cat((pos_data, neg_data), dim=0).to(device)
    indices = torch.randperm(combined_data.size(0))  
    combined_data_shuffled = combined_data[indices]     
    combined_data = combined_data[:100].to(device)
    representation_vectors = model.conv1(combined_data).relu().to(device)
    representation_vectors = model.conv2(representation_vectors).relu().to(device)
    representation_vectors = representation_vectors.view(representation_vectors.shape[0], -1).to(device)
    representation_vectors = model.flatten(representation_vectors).to(device)
    outputs = linear_trans(representation_vectors).to(device)
    _, predicted = torch.max((outputs.data), 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
  print('Accuracy of the network on the testing images: {} %'.format( 100 * correct / total))
