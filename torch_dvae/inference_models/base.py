import abc 
import torch
import torch.nn as nn 

class InferenceBase(nn.Module, metaclass=abc.ABCMeta):
    zdim: int 

    @abc.abstractmethod
    def get_dist(self, ys, xs, *args, **kwargs):
        """
        Within the inference class itself the previous latent variables "zs" should be handled.

        e.g. in an RNN model it internally deals with hidden variables which are passed to the next cell 
        after each computation. This internal state could represent z_{1:t-1} and so there is no need to 
        input it in this method.

        However, if an MLP is used then perhaps consider adding another method to initialize z0 and 
        pass it in this function as a kwarg or arg. 

        Inputs: 
            ys (tensor): noncausal target signal (y_{1:T}) represetation, size (bs, seq, hdim) 
            xs (tensor): noncausal input signal (x_{1:T}) representation, size (bs, seq, hdim) (optional)

        Outputs:
            dist_zs (tensor): inference distributions of the latent sequences 
        """
