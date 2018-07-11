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
    
 # Implement Experience Replay with a new class   
class ReplayMemory(object):
     
     # Capacity is the number of state(s) with some experience
     def __init__(self, capacity):
         self.capacity = capacity
         # List that contains some last 100 events
         self.memory = []
    
    # This function pushes the events into the memory List
     def push(self, event):
         self.memory.append(event)
         # If memory exceeds capacity then delete the oldest event
         if len(self.memory) > self.capacity:
             del self.memory[0]
             
    # determing each batch size(s) 
     def sample(self, batch_size):
         samples = zip(*random.sample(self.memory, batch_size))
         return map(lambda x: Variable(torch.cat(x, 0)), samples)
 
# Implementation of Deep Q Learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        # Reward window to evalute agent 
        self.reward_window = []
        # Create NN
        self.model = Network(input_size, nb_action)
        # Create a new memory
        self.memory = ReplayMemory(100000)
        # Get the Optimizer from PyTorch's adam class
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # vector of 5 dimensios initialized as a tensor with also a fake dimension
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # last action
        self.last_action = 0
        # last reward
        self.last_reward = 0
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*7) # temp = 7
        # Random draw of probs
        action = probs.multinomial()
        return action.data[0,0]
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # We want to get it tensors not in batch(s)
        outputs = self.model(batch_state).gather(1, batch_action).unsqueeze(1).squeeze(1)
        # Get the max q-values of the next state
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        # Compute the loss
        td_loss = F.smooth_l1_loss(outputs, target)
        # To backpropagate and also apply stochastic gradient descent
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        # Update the weights
        self.optimizer.step()
        
    # Update the elements and keep transitions going
    def update(self, reward, new_signal):
        # new state composed with the 3 signals 
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # update the memory 
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)])), torch.Tensor([self.last_reward]))
        # play an action
        action=  self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    # compute the mean of all rewards in the reward_window
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    # save the model 
    def save(self):
        # save the most recently used weights
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict,
                    }, 'last_brain.pth')
    
    # load function
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint ...")
            checkpoint = torch.load('last_brain.pth')
            # update the model
            self.model.load_state_dict(checkpoint['state_dict'])
            # update the parameters
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done !')
        else:
            print("No checkpoint found ...")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        