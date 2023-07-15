import torch 
import torch.nn as nn 

from torch_dvae.measurement_models.base import MeasureBase
from torch_dvae.utils import GaussianNet

class MLPMeasure(MeasureBase):
    def __init__(self, measure_dim: int, zdim: int, hdim: int, ydim: int, xdim: int):
        """
        models measurement mapping, p(y_t|y_{1:t-1}, z_{1:t}, x_{1:T}) 
        if any of these inputs are not be be used, set the corresponding dim. to zero 
        e.g. if we assume y_{1:t-1} is not an input then set ydim = 0 

        *Note measure dim is the actual dimension of the measurements, while ydim could be 
        the dimension of the encoded hidden variable that represents y_{1:t-1}
        """
        super().__init__()
        self.zdim = zdim 
        self.xdim = xdim
        self.ydim = ydim

        self.net = GaussianNet(ydim+zdim+xdim, hdim, measure_dim)

    def get_dist(self, zs, ys, xs):
        """
        Make sure "xs" and "ys" are encoded before inputting into this method. 
        """
        if self.ydim == 0:
            ys = None 
        if self.xdim == 0:
            xs = None 
        if self.zdim == 0:
            zs = None 

        ys_dist = self.net(ys, zs, xs)
        return ys_dist 

