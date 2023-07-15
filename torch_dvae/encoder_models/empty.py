import torch
import torch.nn as nn 
from torch_dvae.encoder_models.base import EncoderBase

class EmptyEncoder(EncoderBase):
    def __init__(self, xdim=0, hdim=0):
        """
        If ys or xs are not used, use the EmptyEncoder class as it outputs "None" when encoding 
        the "sequence" and the GaussianNet and GaussianRNN ignores any None inputs so they aren't accounted for.

        EmptyEncoder also ensures the encoded dim = 0, so any networks that require the input dims. to be 
        specified won't be affected if the EmptyEncoder hdim is used.
        """
        super().__init__()
        self.xdim = 0
        self.hdim = 0

    def encode_fwd(self, xs: torch.Tensor):
        return None

    def encode_bwd(self, xs: torch.Tensor, hs: torch.Tensor):
        return None

    def encode(self, x: torch.Tensor, h):
        return None 
