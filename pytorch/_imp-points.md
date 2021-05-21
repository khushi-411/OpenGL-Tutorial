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
