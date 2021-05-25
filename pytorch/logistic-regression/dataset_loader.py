import torch
import torchvision
import torchvision.transforms as transforms

batch_size=100


def load_data():
    
    try:
        
        train_data = torchvision.datasets.MNIST(root="../../data",
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=True)
        
        test_data = torchvision.datasets.MNIST(root="../../data",
                                               train=False,
                                               transform=transforms.ToTensor()
                                               )
        
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True
                                                   )
        
        test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                          batch_size=batch_size, 
                                          shuffle=False)
        
        
        return train_loader, test_loader
        
    except FileNotFoundError as error:
        print("File Not Found Error during loading data")
        raise error

    except Exception as error:
        print("Exception Error: load_data")
        print(error)
        raise error