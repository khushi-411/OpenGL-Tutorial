import torch.nn as nn



class ConvNet(nn.Module):

    def __init__(self, _output_size, _kernel_size, _stride, _padding):
        
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, _kernel_size, _stride, _padding),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, _kernel_size, _stride, _padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Linear(7*7*32, _output_size)
        

    def forward(self, x):
        
        _output = self.layer1(x)
        _output = self.layer2(_output)
        _output = _output.reshape(_output.size(0), -1)
        _output = self.fc(_output)

        return _output
