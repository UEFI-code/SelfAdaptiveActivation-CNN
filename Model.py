import numpy as np
import torch
import torchvision
from torch import nn,optim
class myModel(nn.Module):
    def __init__(self):
        super(myModel,self).__init__()
        self.layera = nn.Sequential(
            nn.Conv2d(3,16,3), #in_channels out_channels kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 2,stride = 2)  #149
        )
        self.layerb = nn.Sequential(
            nn.Conv2d(16,64,3,2),  #74    #
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size =2,stride=2)  #37

        )
        self.layerc = nn.Sequential(
            nn.Conv2d(64,64,3,2),  #18
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 2, stride = 2)  #9
        )
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax()
        self.li0 = nn.Linear(576,1024)
        self.li1 = nn.Linear(1024,512)
        self.li2 = nn.Linear(512,256)
        self.li3 = nn.Linear(256,4)
        self.optim0a = nn.Linear(512,1024)
        self.optim0b = nn.Linear(512,1024)
        self.optim0c = nn.Linear(512,1024)
        self.optim0d = nn.Linear(512,1024)
        self.optim0e = nn.Linear(512,1024)
        self.optim1a = nn.Linear(512,512)
        self.optim1b = nn.Linear(512,512)
        self.optim1c = nn.Linear(512,512)
        self.optim1d = nn.Linear(512,512)
        self.optim1e = nn.Linear(512,512)
        self.optim2a = nn.Linear(512,256)
        self.optim2b = nn.Linear(512,256)
        self.optim2c = nn.Linear(512,256)
        self.optim2d = nn.Linear(512,256)
        self.optim2e = nn.Linear(512,256)
        #self.fc3optim = nn.Linear(1,4)
    def forward(self, x):
        x = self.layera(x)
        x = self.layerb(x)
        x = self.layerc(x)
        x = x.view(x.size(0),-1)
        ones = torch.rand(x.size(0),512).cuda()
        x = self.li0(x)
        cacheA = self.optim0a(ones)
        cacheB = self.optim0b(ones)
        cacheC = self.optim0c(ones)
        cacheD = self.optim0d(ones)
        cacheE = self.optim0e(ones)
        x = self.relu(x) * cacheA + cacheB + self.sigmoid(x) * cacheE
        x = self.li1(x)
        cacheA = self.optim1a(ones)
        cacheB = self.optim1b(ones)
        cacheC = self.optim1c(ones)
        cacheD = self.optim1d(ones)
        cacheE = self.optim1e(ones)
        x = self.relu(x) * cacheA + cacheB + self.sigmoid(x) * cacheE
        x = self.li2(x)
        cacheA = self.optim2a(ones)
        cacheB = self.optim2b(ones)
        cacheC = self.optim2c(ones)
        cacheD = self.optim2d(ones)
        cacheE = self.optim2e(ones)
        x = self.relu(x) * cacheA + cacheB + self.sigmoid(x) * cacheE
        x = self.li3(x)
        x = self.softmax(x)
        return x

