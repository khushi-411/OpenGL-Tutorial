import torch
import torch.nn as nn
from dataset_loader import load_data
from model import NeuralNetwork



def train(train_loader, _model, _epochs, _input_size, _device, _optimizer, _criterian):
    
    try:
        
        total_step = len(train_loader)
        
        for epoch in range(_epochs):
            
            for i, (_images, _labels) in enumerate(train_loader):
                
                # move tensor to the configured device
                _images = _images.reshape(-1, _input_size).to(_device)
                _labels = _labels.to(_device)
                
                # forward pass
                _outputs = _model(_images)
                _loss = _criterian(_outputs, _labels)
                
                # backward and optimization
                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, _epochs, i+1, total_step, _loss.item()))
                
        
    except:
        print("An error occured while training")
        raise
        
def test(test_loader, _input_size, _model, _device):
    
    try:
        
        with torch.no_grad():
            
            _correct=0
            _total=0
            
            for _images, _labels in test_loader:
                
                _images = _images.reshape(-1, _input_size).to(_device)
                _labels = _labels.to(_device)
                
                _outputs = _model(_images)
                _, _predicted = torch.max(_outputs.data, 1)
                
                _total += _labels.size(0)
                _correct += (_predicted == _labels).sum().item()
                
            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * _correct / _total))
        
    except:
        print("An error occured while testing")
        raise
        
def save_model(_model):
    
    try:
        
        # Save the model checkpoint
        torch.save(_model.state_dict(), 'model.ckpt')
        
    except:
        
        print("An error occred while saving model")
        

def main(_input_size, _output_size, _hidden_size, _learning_rate, _batch_size, _epochs, _device):
    
    try:
        
        # loading dataset
        train_loader, test_loader = load_data()
        
        # loading model
        _model = NeuralNetwork(_input_size, _hidden_size, _output_size).to(_device)
        print(_model)
        print()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(_model.parameters(), lr=_learning_rate)  
        
        # training dataset
        train(train_loader, _model, _epochs, _input_size, _device, optimizer, criterion)
        
        # testing dataset
        test(test_loader, _input_size, _model, _device)
        
        # save model
        save_model(_model)
        
    except:
        
        print("An error Occured!")
        
        
if __name__ == "__main__":
    
    # checking type of device        
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(_device))
    print()
    
    # hyperparameters
    _input_size = 28*28
    _output_size = 10
    _hidden_size = 512
    _learning_rate = 0.001
    _batch_size = 64
    _epochs = 5
    
    # calling main function
    main(_input_size, _output_size, _hidden_size, _learning_rate, _batch_size, _epochs, _device)
    
    