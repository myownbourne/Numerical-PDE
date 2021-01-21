import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time

def phi(x):
    x[x<0]=0
    return x**3

class Model(torch.nn.Module):
    def __init__(self,width):
        super(Model,self).__init__()
        self.layer1=torch.nn.Linear(1, width)
        self.layer2=torch.nn.Linear(width,width)
        self.layer3=torch.nn.Linear(width,width)
        self.layer4=torch.nn.Linear(width,width)
        self.layer5=torch.nn.Linear(width,width)
        self.layer6=torch.nn.Linear(width,1)
    def forward(self,x):
        y=self.layer1(x)
        y=y+phi(self.layer3(phi(self.layer2(y)))) 
        y=y+phi(self.layer5(phi(self.layer4(y))))
        temp=self.layer6(y)
        return x*(1-x)*temp

def f(x):
    return torch.sin(x)/np.sin(1)

width=40
model=Model(width)
optimizer=optim.Adam(model.parameters())
max_iter=100
size=1000

for i in range(max_iter):
    optimizer.zero_grad()
    x=torch.rand(size,1)
    x.requires_grad=True
    ux=model(x)
    #compute loss
    dux=torch.autograd.grad(outputs=ux,inputs=x,grad_outputs=torch.ones(ux.shape),create_graph=True)
    dux2=((dux[0]**2).sum(1)).reshape([len(dux[0]), 1])
    loss=torch.sum(0.5*dux2-f(x)*ux)/len(x)
    
    if i%10==0:
        print('epoch:',i,'loss:', '%2e' % loss.item())
    loss.backward()
    optimizer.step() 

torch.save(model.state_dict(), 'parameters.pkl')