import torch
import torch.nn as nn
from dataset_loader import load_data
import matplotlib.pyplot as plt

_input_size = 1
_output_size = 1
_epochs_num = 100
_learning_rate = 0.0001

class LinearRegressionModel(nn.Module):
  
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=_input_size, out_features=_output_size, bias=True)  
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
print()

_x_train, _y_train = load_data()

_model = LinearRegressionModel().to(device)
print(_model)
print()

criterion = nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(_model.parameters(), lr = _learning_rate)

for _epoch in range(_epochs_num):
    

    # forward
    _y_pred = _model(_x_train)

    # calculate loss
    loss = criterion(_y_pred, _y_train)

    # backward and loss  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if _epoch % 10 == 0:
        print('Epoch- {}, loss- {:.4f}'.format(_epoch, loss.item()))
        
# Plot the graph
predicted = _model(torch.from_numpy(_x_train.numpy())).detach().numpy()
plt.plot(_x_train.numpy(), _y_train.numpy(), 'ro', label='Original data')
plt.plot(_x_train.numpy(), predicted, label='Fitted line')
plt.legend()
plt.show()

        