import torch
import torch.nn as nn 

from torch_dvae.utils import GaussianNet
from torch_dvae.initializers.base import InitializeBase

class MeasureInitializer(InitializeBase):
    def __init__(self, hdim:int, zdim: int, ydim: int, xdim: int):
        super().__init__()
        self.ydim = ydim 
        self.xdim = 0  

        self.net = GaussianNet(ydim, hdim, zdim)
    
    def forward(self, ys, xs):
        y = ys[:,0,:]

        z0_dist = self.net(y)
        return z0_dist 