import torch
import torch.nn as nn

# hyperparameters

input_size = 28 * 28 # 784
output_size = 10


class LogisticRegressionModel(nn.Module):
  
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size, bias=True)  
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    