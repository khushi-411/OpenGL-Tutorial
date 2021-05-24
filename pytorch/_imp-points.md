#### **Working with Dataset**

* `torch.utils.data.Dataset`: Stores samples and corresponding labels
* `torch.utils.data.DataLoader`: wraps an iterable

#### **Creating Models**

* `__init__`: define layers of network
* `forward`: specify how it will move

```
**Note:** to(device)- specifies which device to use for tensors in pytorch
```

#### **Loss Function & Optimization**

`torch.optim`: to implement various optimization algorithm----------hold current value, then update parameter based on computed gradients

#### **Training & Testing**

#### **Save Model**

1) Saving and Loading Model

`torch.save(model.state_dict(), "model.pth")`: Store the learned parameters as an internal state dictionary

To load the model weights ------ i) Create the instance of the same model, ii) Load parameters using `load_state_dict()` method

```
model.load_state_dict(torch.load('model.pth'))
model.eval()
```
> *Note:* Call `model.eval()` before calling dropout and normalization layers

2) Saving and Loading Model with Shapes

Used when we want to first save structure of class (class defines structure of network). This module uses python `pickel` module.

```
torch.save(model, 'model.pth')
model = torch.load('model.pth')
```

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### **Datasets and DataLoaders**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### **Tansforms**

`transform`: To modify the features.

`traget_transform`: To modify the labels.

Example: FashionMNIST Datasets

- features are in PIL image format ----------- Required as normalized tensors
- labels are integers ----------- Required as one-hot encoded tensors

Therefore use:

- `ToTensor()`: converts a PIL image or NumPy ndarray into a FloatTensor...........Scales image intensity pixels in range [0., 1.]
- `Lambda Transforms`: They are user definid lambda function which converts integer values to one-hot-encoded tensors

```
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


### **Neural Networks**

`torch.nn`: Used to create neural networks

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

In a forward pass, autograd does two things simultaneously:

* run the requested operation to compute a resulting tensor, and
* maintain the operation’s gradient function in the DAG (directed acyclic graph).

```
prediction = model(data)
```

##### **Backward Pass**

**Calculating loss:** Using model prediction and original label

The backward pass kicks off when `.backward()` is called on the DAG root. autograd then:

* computes the gradients from each `.grad_fn`,
* accumulates them in the respective tensor’s `.grad` attribute, and
* using the chain rule, propagates all the way to the leaf tensors.

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

##### `frozen parameters`

Parameters that don’t compute gradients

Why freeze?

- most of the model and typically only modify the classifier layers to make predictions on new labels. 

##### **Finetuning the model**

Model - 10 labels

Classifier in last layer - `model.fc`

Last layer - Replacing it with new linear layer (unfrozen by default)



```
model.fc = nn.Linear(512, 10)
```

We find that all the layers are frozen except the last layer ( `model.fc` ). Only parameters of last layer are used to compute weights and bias `gradients`.

```
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```


