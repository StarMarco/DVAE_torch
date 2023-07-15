import abc 
import torch
import torch.nn as nn 

class EncoderBase(nn.Module, metaclass=abc.ABCMeta):
    xdim: int  
    hdim: int 

    @abc.abstractmethod
    def encode_fwd(self, xs: torch.Tensor):
        """
        Encodes sequences x_{1:T}, where h_t := x_{1:t-1}

        Inputs: 
            xs (tensor): input sensor signal, size (bs, seq, xdim)

        Outputs:
            hs (tensor): represents an entire sequence of inputs before time "t", size (bs, seq, hdim)
        """

    @abc.abstractmethod
    def encode_bwd(self, xs: torch.Tensor, hs):
        """
        Encodes sequences x_{1:T}, where h_t := x_{1:T}

        Inputs: 
            xs (tensor): input sensor signal, size (bs, seq, xdim)
            hs (tensor): represents a sequence of inputs x_{1:t-1}, size (bs, seq, hdim)

        Outputs:
            hs (tensor): represents an entire sequence of inputs in a noncausal manner, size (bs, seq, hdim)
        """ 

    @abc.abstractmethod
    def encode(self, x, h):
        """
        Real time encoder 

        Inputs:
            x (tensor): current input variable, size (bs, xdim)
            h (tensor): representation of previous variables, size (bs, hdim)

        Outputs:
            h (tensor): representation of current sequence 
        """