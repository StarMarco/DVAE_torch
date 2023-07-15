import abc 
import torch
import torch.nn as nn 

from torch_dvae.encoder_models.base import EncoderBase

class InitializeBase(nn.Module, metaclass=abc.ABCMeta):
    x_encoder: EncoderBase
    y_encoder: EncoderBase
    
    @abc.abstractmethod
    def forward(self, y, x):
        """
        Returns initial latent variable z0 distribution

        Inputs: 
            y (tensor): observation
            x (tensor): input

        Outputs:
            dist_zs (tensor): the distributions of the latent variables after applying the transition dynamics, size (bs, hdim)
        """
