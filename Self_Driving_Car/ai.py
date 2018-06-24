# AI for Self Driving Car

# Importing the libraries

# For arrays
import numpy as np
import random

# To load the model
import os

# For dynamic graph calculations 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# From tensors to variables with gradients
import torch.autograd as autograd
from torch.autograd import Variable

# Create the architecture of the Neural Network

# Rectifier function is used here as activation function
# There are 5 q-values

# Inherit from nn.Module class as it containes essential tools for a NN
class Network(nn.Module):
    
    # input_size is the number of inputs
    # nb_action is the output 3 actions - forward, right or left
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        # 2 Full connections
        # 30 neurons in the hidden layer forming a linear connection
        
        # Connection from input to hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        
        # Connection from hidden layer to output layer
        self.fc2 = nn.linear(30, nb_action)
        
    # Function that performs forward propagation
    def forward(self, state):
        
        # Activate the hidden neurons
        x = F.relu(self.fc1(state))
        # Get the q-values
        q_values = self.fc2(x)
        return q_values