import torch 
import torch.nn as nn 

from torch_dvae.transition_models.base import TransitionBase
from torch_dvae.utils import GaussianNet

class MLPTransition(TransitionBase):
    def __init__(self, zdim: int, hdim: int, ydim: int, xdim: int):
        """
        models transition p(z_t|z_{1:t-1}, y_{1:t-1}, x_{1:T}). 
        if any of these inputs are not be be used, set the corresponding dim. to zero 
        e.g. if we assume y_{1:t-1} is not an input then set ydim = 0 
        """
        super().__init__()
        self.zdim = zdim 
        self.ydim = ydim 
        self.xdim = xdim 

        self.net = GaussianNet(zdim+ydim+xdim, hdim, zdim)

    def get_dist(self, zs, ys, xs):
        """
        Make sure "xs" and "ys" are encoded before inputting into this method 
        """
        if self.ydim == 0:
            ys = None 
        if self.xdim == 0:
            xs = None 
        if self.zdim == 0:
            zs = None 

        zs_dist = self.net(zs, ys, xs)
        return zs_dist 

