import abc 
import torch
import torch.nn as nn 

class MeasureBase(nn.Module, metaclass=abc.ABCMeta):
    zdim: int 
    ydim: int 

    @abc.abstractmethod
    def get_dist(self, zs: torch.Tensor, ys, xs):
        """
        Encodes sequences x_{1:T}, where h_t := x_{1:t-1}

        Inputs: 
            zs (tensor): latent variable (z_{1:t}) representation, size (bs, seq, zdim)
            ys (tensor): previous observations (y_{1:t-1}) representation, size (bs, seq, hdim) (optional)
            xs (tensor): noncausal input signal (x_{1:T}) representation, size (bs, seq, hdim) (optional)

        Outputs:
            dist_ys (tensor): the distributions of the observations, size (bs, seq, hdim)
        """
