import torch
import torch.nn as nn
import torchvision
from model import LogisticRegressionModel
from dataset_loader import load_data


# Hyper-parameters 
input_size = 28 * 28    # 784
output_size = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

def train(train_loader, input_size, num_epochs):
    
    try:
    
        total_step = len(train_loader)
        
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Reshape images to (batch_size, input_size)
                images = images.reshape(-1, input_size)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    except:
        print("An error occured while training")
        
        
def test(test_loader):
    
    try:

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, input_size)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        
            print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
            
    except:
        print("An error occured while testing")
        
def save_model(model):
    
    try:
        
        # Save the model checkpoint
        torch.save(model.state_dict(), 'model.ckpt')
        
    except:
        print("An Error Occured")
        

# checking type of device        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
print()

# loading dataset      
train_loader, test_loader = load_data()

# linear model
model = LogisticRegressionModel().to(device)
print(model)
print()

# loss function and optimization
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# training & testing
train(train_loader, input_size, num_epochs)
test(test_loader)

# saving model
save_model(model)

