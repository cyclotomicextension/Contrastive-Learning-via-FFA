import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the positive examples
x_pos = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y_pos = x_pos ** 2

# Define the negative examples (corrupted x values)
x_neg = x_pos + 0.1 * torch.randn(x_pos.size())
y_neg = x_neg ** 2

# Concatenate the positive and negative examples
x = torch.cat([x_pos, x_neg], dim=0)
y = torch.cat([torch.ones(x_pos.size(0)), torch.zeros(x_neg.size(0))], dim=0)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create an instance of the network
net = Net()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Train the network
for epoch in range(10000):
    # Forward pass
    output = net(x)
    y_pred=output[:,0]
    # Compute the loss
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {loss.item()}")

# Evaluate the network
with torch.no_grad():
    output = net(x)
    predicted = output > 0.5
    accuracy = (predicted == y.byte()).sum().item() / len(y)
    print(f"Accuracy: {accuracy}")
