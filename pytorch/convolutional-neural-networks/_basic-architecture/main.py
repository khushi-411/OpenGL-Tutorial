import torch
import torch.nn as nn
from dataset_loader import load_data
from model import ConvNet



def train(_train_loader, _model, _num_epochs, _device, _criterion, _optimizer):
    
    try:
        
        _total_steps = len(_train_loader)
        
        for _epochs in range(_num_epochs):
            
            for i, (_images, _labels) in enumerate(_train_loader):
                
                _images = _images.to(_device)
                _labels = _labels.to(_device)
                
                # forward pass
                _outputs = _model(_images)
                _loss = _criterion(_outputs, _labels)
            
                # backward pass and optimization
                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(_epochs+1, _num_epochs, i+1, _total_steps, _loss.item()))
        
    except Exception as error:
        print("An error occured while training")
        print(error)
        raise error
        
        
def test(_test_loader, _model, _device):
    
    try:
        
        with torch.no_grad():
            
            _correct=0
            _total=0
            
            for _images, _labels in _test_loader:
                
                _images = _images.to(_device)
                _labels = _labels.to(_device)
                
                _outputs = _model(_images)
                _, _predicted = torch.max(_outputs.data, 1)
                
                _total += _labels.size(0)
                _correct += (_predicted == _labels).sum().item()
                
            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * _correct / _total))
        
    except Exception as error:
        print("An error occured while testing")
        print(error)
        raise error
        

def save_model(_model):
    
    try:
        
        # Save the model checkpoint
        torch.save(_model.state_dict(), 'model.ckpt')
        
    except:
        
        print("An error occred while saving model")
        
def main(_output_size, _num_epochs, _batch_size, _learning_rate, _kernel_size, _stride, _padding, _device):
    
    try:
        
        # loading dataset
        _train_loader, _test_loader = load_data()

        # model instance
        _model = ConvNet(_output_size, _kernel_size, _stride, _padding).to(_device)
        print(_model)
        
        # Loss and optimizer
        _criterion = nn.CrossEntropyLoss()
        _optimizer = torch.optim.Adam(_model.parameters(), lr=_learning_rate)

        # training
        train(_train_loader, _model, _num_epochs, _device, _criterion, _optimizer)

        # testing
        test(_test_loader, _model, _device)

        # saving model
        save_model(_model)
        
        
    except Exception as error:
        print("An Error Occured!")
        print(error)
        raise error
        
        
if __name__ == "__main__":
    
    
    # hyperparameters    
    _output_size = 10
    _num_epochs = 5
    _batch_size = 100
    _learning_rate = 0.001
    _kernel_size = 5
    _stride = 1
    _padding = 2
    
    
    # checking type of device  
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(_device))
    print()
    
    main(_output_size, _num_epochs, _batch_size, _learning_rate, _kernel_size, _stride, _padding, _device)
    