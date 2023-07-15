import torch
import torch.nn as nn 

from torch_dvae.utils import GaussianNet
from torch_dvae.initializers.base import InitializeBase
from torch_dvae.encoder_models import EmptyEncoder, InitEncoder

class ControlInitializer(InitializeBase):
    def __init__(self, hdim:int, zdim: int, ydim: int, xdim: int):
        super().__init__()
        self.ydim = 0  
        self.xdim = xdim   # returns xdim 

        self.net = GaussianNet(xdim, hdim, zdim)
    
    def forward(self, ys, xs):
        x = xs[:,0,:]

        z0_dist = self.net(x)
        return z0_dist 