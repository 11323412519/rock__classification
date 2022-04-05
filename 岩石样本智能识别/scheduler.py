import torch
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR


#余弦退火
def CosineannealingLR(optimizer):
    return CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

def CosineannealingWarmRestarts(optimizer):
    return CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)

#指数衰减
def ExponentialLR(optimizer,gamma=0.98):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

#固定步长衰减
def StepLR(optimizer,step_size=30,gamma=0.65):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def MultiStepLR(optimizer,milestones,gamma=0.1):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=gamma)