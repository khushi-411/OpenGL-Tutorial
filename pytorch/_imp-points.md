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


