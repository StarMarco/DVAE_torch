import torch
import torch.nn as nn 
from torch_dvae.encoder_models.base import EncoderBase

class InitEncoder(EncoderBase):
    def __init__(self, xdim, hdim=0):
        """
        Handy for initializers, simply returns the sequence when using encode_fwd
        so the initializer network can extract the first element of the sequence 
        """
        super().__init__()
        self.xdim = xdim
        self.hdim = xdim

    def encode_fwd(self, xs: torch.Tensor):
        return xs

    def encode_bwd(self, xs: torch.Tensor, hs: torch.Tensor):
        return None

    def encode(self, x, h):
        return None 
