import torch
import torch.nn as nn


_input_size = 28*28
_hidden_size = 512 
_output_size = 10


class NeuralNetwork(nn.Module):
    
    def __init__(self, _input_size, _hidden_size, _output_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(_input_size, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _output_size),
            nn.ReLU()
            )
    
    def forward(self, x):
        x = self.flatten(x)
        _output = self.linear_relu_stack(x)
        return _output
    