import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from math import *
import time

def activation(x):
    return x*torch.sigmoid(x)

class Net(torch.nn.Module):
    def __init__(self,input_width,layer_width):
        super(Net,self).__init__()
        self.layer_in=torch.nn.Linear(input_width,layer_width)
        self.layer1=torch.nn.Linear(layer_width,layer_width)
        self.layer2=torch.nn.Linear(layer_width,layer_width)
        self.layer3=torch.nn.Linear(layer_width,layer_width)
        self.layer4=torch.nn.Linear(layer_width,layer_width)
        self.layer5=torch.nn.Linear(layer_width,layer_width)
        self.layer6=torch.nn.Linear(layer_width,layer_width)
        self.layer_out=torch.nn.Linear(layer_width,1)
    def forward(self,x):
        y=self.layer_in(x)
        y=y+activation(self.layer2(activation(self.layer1(y))))
        y=y+activation(self.layer4(activation(self.layer3(y))))
        y=y+activation(self.layer6(activation(self.layer5(y))))
        output=self.layer_out(y)
        return output

dimension=1
input_width,layer_width=dimension,20
net=Net(input_width,layer_width)

def u_ex(x):
    temp=1
    for i in range(dimension):
        temp=temp*torch.sin(pi*x[:,i])
    u_temp=1.0*temp
    return u_temp.reshape([x.size()[0],1])

def f(x):
    temp = 1.0
    for i in range(dimension):
        temp = temp * torch.sin(pi*x[:, i])
    u_temp = 1.0 * temp
    f_temp = dimension * pi**2 * u_temp 
    return f_temp.reshape([x.size()[0],1])

def generate_sample(data_size):
    sample_temp=torch.rand(data_size,dimension)
    return sample_temp

def model(x):
    x_temp = x
    D_x_0 = torch.prod(x_temp, axis = 1).reshape([x.size()[0], 1]) 
    D_x_1 = torch.prod(1.0 - x_temp, axis = 1).reshape([x.size()[0], 1]) 
    model_u_temp = D_x_0 * D_x_1 * net(x)
    return model_u_temp.reshape([x.size()[0], 1])

x=torch.tensor([[1],[2],[3]])
print(torch.prod(1.0 - x,1).reshape([x.size()[0], 1]))
