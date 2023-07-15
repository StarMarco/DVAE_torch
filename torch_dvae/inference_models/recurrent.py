import torch 
import torch.nn as nn 

from torch_dvae.inference_models.base import InferenceBase
from torch_dvae.encoder_models.base import EncoderBase
from torch_dvae.utils import GaussianRNN

class RNNInference(InferenceBase):
    def __init__(self, zdim: int, hdim: int, ydim: int, xdim: int):
        super().__init__()
        self.zdim = zdim 
        self.ydim = ydim 
        self.xdim = xdim

        self.inf_net = GaussianRNN(ydim + xdim, hdim, zdim) # zs is incorporated in the inputs through the RNNs hidden dim 

    def get_dist(self, ys, xs=None):
        """
        Make sure "ys" and "xs" are representations of the sequences (i.e. make sure they are 
        outputs of your encoders before inputting into this method)
        """
        if self.ydim == 0:
            ys = None 
        if self.xdim == 0:
            xs = None 

        dist_zs = self.inf_net(ys, xs)
        return dist_zs 
    
