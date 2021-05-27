import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self, _output_size):
        
        super(LeNet5, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh()
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh()
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
            )
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84, bias=True),
            nn.Tanh()
            )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=84, out_features=_output_size, bias=True),
            nn.Softmax()
            )
        

    def forward(self, x):
        
        _output = self.layer1(x)
        _output = self.layer2(_output)
        _output = self.layer3(_output)
        _output = _output.reshape(_output.size(0), -1)
        _output = self.fc1(_output)
        _output = self.fc2(_output)

        return _output
