import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data import DataLoader
from torch.nn import DataParallel

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Example architecture: Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Example architecture: Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        # Example forward pass
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize  model
model = MyModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Check if distributed training is enabled
if dist.is_available() and dist.is_initialized():
    # DDP: Wrap the model with DistributedDataParallel
    model = DistributedDataParallel(model)
else:
    # Single GPU or CPU training
    model = DataParallel(model)

# Optionally, FSDP implementation
from fsdp import FullyShardedDataParallel

if torch.cuda.is_available():
    model = FullyShardedDataParallel(model)

# Training loop
def train_epoch(dataloader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss

# Main training loop
def train():
    num_epochs = 10

    # Assuming dataloader is defined and ready
    for epoch in range(num_epochs):
        total_loss = train_epoch(dataloader, model, criterion, optimizer, device)

        if dist.is_available() and dist.is_initialized():
            # DDP: Sum losses across all processes
            dist.all_reduce(total_loss)

        if dist.is_available() and dist.is_initialized():
            # Print average loss per GPU in DDP
            print(f"Epoch {epoch}, Loss: {total_loss / dist.get_world_size()}")
        else:
            # Single GPU or CPU
            print(f"Epoch {epoch}, Loss: {total_loss}")

# Run the training loop
train()
