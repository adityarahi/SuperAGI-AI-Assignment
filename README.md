# SuperAGI-AI-Assignment

# Training Loop with Single GPU, DDP, and FSDP Support

## Overview

This Python script provides a flexible training loop that supports single GPU training, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP) in PyTorch. The script includes a simple example of a neural network model (`MyModel`) with convolutional and fully connected layers.

## Files

- **q3.py**: Main Python script containing the training loop and model definition.

## Prerequisites

- Python 3.x
- PyTorch
- FSDP (if using Fully Sharded Data Parallel)

```bash
pip install torch
pip install fsdp  # Only required if using FSDP

# Usage
Run the training script:

python q3.py

MyModel Architecture
Model Architecture
The MyModel class defines a simple neural network architecture with convolutional and fully connected layers.

Convolutional Layers
Conv1: Input channels - 3, Output channels - 64, Kernel size - 3x3, ReLU activation, Max pooling (2x2)
Conv2: Input channels - 64, Output channels - 128, Kernel size - 3x3, ReLU activation, Max pooling (2x2)
Fully Connected Layers
Flatten: Flatten the output from convolutional layers
FC1: Input size - 128 * 8 * 8, Output size - 512, ReLU activation
FC2: Input size - 512, Output size - 10 (assuming 10 classes for classification)

#Usage
# Initialize MyModel
model = MyModel()

# Forward pass example
input_data = torch.randn((batch_size, 3, 64, 64))  # Example input tensor
output = model(input_data)

## Training Loop
# Script Structure
The script initializes MyModel, defines a loss function (CrossEntropyLoss), and sets up an optimizer (SGD).
It checks for the availability of multiple GPUs and applies DDP or FSDP accordingly.
The training loop (train_epoch function) iterates over the dataset, performs forward and backward passes, and updates the model parameters.
The main training loop (train function) executes the training for a specified number of epochs.
Running the Script
Run the script using the command: python train.py
Training Modes
Single GPU: The script automatically detects a single GPU setup and applies DataParallel.
DDP: If multiple GPUs are available and DDP is initialized, the script uses DistributedDataParallel.
FSDP: Optionally, if FSDP is installed and GPUs are available, the script initializes FullyShardedDataParallel.
Acknowledgments
The code template is based on the PyTorch DDP tutorial and FSDP documentation.

