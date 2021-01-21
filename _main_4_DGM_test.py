import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time
import matplotlib.pyplot as plt

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

width=40
model=Model(width)
model.load_state_dict(torch.load('parameters.pkl'))
eps=1e-5
x=torch.arange(0,1+eps,0.01)
x=x.reshape([len(x),1])
y=model(x)
y_true=np.sin(x)/np.sin(1)-x

x=x.detach().numpy()
y=y.detach().numpy()
y_true=y_true.detach().numpy()

fig=plt.figure()
fig.suptitle('Visualization of Deep Ritz method')
ax1=fig.add_subplot(1,2,1) 
ax2=fig.add_subplot(1,2,2)

ax1.plot(x,y,'blue')
ax2.plot(x,y_true,'red')
ax1.set_title('Numerical results by NN')
ax2.set_title('objtective function')
plt.show()