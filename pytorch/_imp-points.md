#### **Working with Dataset**

* `torch.utils.data.Dataset`: Stores samples and corresponding labels
* `torch.utils.data.DataLoader`: wraps an iterable

#### **Creating Models**

* `__init__`: define layers of network
* `forward`: specify how it will move

```
**Note:** to(device)- specifies ehich device to use for tensors in pytorch
```

#### **Loss Function & Optimization**

`torch.optim`: to implement various optimization algorithm----------hold current value, then update parameter based on computed gradients

#### **Training & Testing**

#### **Save Model**: `torch.save(model.state_dict(), "model.pth")`


### **Neural Networks**

`torch.autograd`: automatic differentiation engine

#### **Training Neural Networks**

* `Forward Propagation`: Runs input in the given equation and guess the best correct output.
* `Backward Propagation`: Traverse backward from the output, collecting derivative of error (wrt to parameters of functions--- gradients) & optimizing the parameters


#### **Example**

* From `torchvision` -------- Loading pretrained model ---- **resnet18()**
* Created `random data tensor` -------- Represent Image (3-channels, height-64, width-64) 
* `label`: Initialised to some random values

```
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
```

##### **Forward Pass**

Running input through each layer

```
prediction = model(data)
```

##### **Backward Pass**

**Calculating loss:** Using model prediction and original label

```
loss = (prediction - labels).sum()
```

**Backpropagation:** Called on `error tensor`. Here, loss.

```
loss.backward() 
```

Then `autograd` calculates and stores the gradients for each model parameter in `grad` attribute.

##### **Loading Optimizer**

Here, SGD.

```
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

Finally, we call `.step()` to initiate gradient descent. 

```
optim.step()
```
