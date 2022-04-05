from torch import optim
import torch
def SGD(model,lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer
def RMSprop(model,lr):
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    return optimizer
def Adam(model,lr):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    return optimizer