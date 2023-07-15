import torch
import torch.nn as nn 
from torch_dvae.encoder_models.base import EncoderBase

class RNNEncoder(EncoderBase):
    def __init__(self, xdim, hdim):
        super().__init__()
        self.xdim = xdim
        self.hdim = hdim

        self.rnn_fwd = nn.GRU(xdim, hdim, batch_first=True)
        self.rnn_bwd = nn.GRU(xdim+hdim, hdim, batch_first=True)

    def encode_fwd(self, xs: torch.Tensor):
        bs, _, xdim = xs.shape
        x0 = torch.zeros([bs, 1, xdim]).to(xs.device)
        x = torch.cat([x0, xs[:,:-1,:]], dim=1)

        hs, _ = self.rnn_fwd(x)
        return hs

    def encode_bwd(self, xs: torch.Tensor, hs: torch.Tensor):
        hxs = torch.cat([hs, xs], dim=-1)

        g_revs, _ = self.rnn_bwd(torch.flip(hxs, [1]))    # backward RNN 
        gs = torch.flip(g_revs, [1])
        return gs

    def encode(self, x: torch.Tensor, h: torch.Tensor):
        x = x.unsqueeze(1)  # (bs, 1, xdim)
        h = h.unsqueeze(0)  # (1, bs, hdim)
        
        hs, _ = self.rnn_fwd(x, h)
        return hs[:,0,:]

