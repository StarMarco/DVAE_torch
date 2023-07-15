import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as dists

def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = F.softplus
  elif name == 'selu':
    nl = F.selu
  elif name == 'elu':
    nl = F.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl

def score_func(x, t):
    '''
    Score function from NASA C-MAPSS data pdf (comes with C-MAPSS dataset download)
    *Note bs = 1 in testing (ONLY USE THIS FUNCTION IN TESTING)

    INPUTS: 
        x (tensor): RUL estimates 
        t (tensor): True RUL values 
    
    OUTPUTS: 
        score (tensor):
            when,
                e = x-t < 0, score = exp(-e/13) - 1
                e = x-t > 0, score = exp(e/10) - 1

        The negetive values are penalized less then the postive values. 
        Hence, when e < 0 true RUL is higher than the estimate so our estimate 
        says the component will fail before it actually will (estimate is conservative)
        so it is less penalized. 
    '''

    error = (x - t)       # error = estimated RUL - True RUL
    error_less = (error < 0) * error
    error_more = (error >= 0) * error

    score_less = torch.exp(-error_less / 13) - 1
    score_more = torch.exp(error_more / 10) - 1 

    score = score_less + score_more
    return score 

def alpha_coverage(ys, y, p=0.95):
    """
    Counting the amount of points within the confidence interval (CI). The closer the number is to the CI 
    the better e.g. if we take the mean of these alpha_coverages and get 0.95 for a 95% CI then this is ideal.

    Found in the paper from Mitici, de Pater, Barros, Zeng (2023),
    "Dynamic predictive maintenance for multiple components using data-driven probabilist RUL prognostics:
    The case of turbofan engines"

    Inputs: 
        ys (tensor): The samples that make up the distribution, size (N, seq, dim)
        y (tensor): actual target, size (seq, dim)
        p (float): percentage the CI should cover (between 0-1)

    Outputs:
        alpha_coverage: a metric which counts if the target falls within the 
            confidence bounds, size (seq, dim)
    """
    assert 0 <= p and p <= 1, "WARNING: alpha_coverage() method requires a 'p' value between 0 and 1"
    diff = (1-p) / 2.
    lower_bo = torch.quantile(ys, 1-p - diff, dim=0)
    upper_bo = torch.quantile(ys, p + diff, dim=0)

    l = (lower_bo <= y).int()
    u = (y <= upper_bo).int()
    bound = l + u

    coverage = (bound == 2).float()
    return coverage 

def alpha_mean(ys, p=0.95):
    """
    The width of the confidence intervals 
    also found in the paper from Mitici, de Pater, Barros, Zeng (2023),
    "Dynamic predictive maintenance for multiple components using data-driven probabilist RUL prognostics:
    The case of turbofan engines"

    Inputs: 
        ys (tensor): The samples that make up the distribution, size (N, seq, dim)
        p (float): percentage the CI should cover 
    Outputs:
        alpha_mean: a metric for the width of the confidence intervals, size (seq, dim)
    """
    assert 0 <= p and p <= 1, "WARNING: alpha_mean() method requires a 'p' value between 0 and 1"
    diff = (1-p) / 2.
    lower_bo = torch.quantile(ys, 1-p - diff, dim=0)
    upper_bo = torch.quantile(ys, p + diff, dim=0)

    return upper_bo - lower_bo

# -------------------
# Neural Networks
# -------------------
class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = nn.Linear(hidden_dim, output_dim)

    for l in [self.linear1, self.linear2, self.linear3]:
      nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity(self.linear1(x))
    h = h + self.nonlinearity(self.linear2(h))
    return self.linear3(h)

class GaussianNet(nn.Module):
    def __init__(self, xdim, hdim, zdim):
        super().__init__()
        self.zdim = zdim 
        self.net = MLP(xdim, hdim, zdim*2)

    def get_stats(self, *args):
        tensor_list = [i for i in args if i is not None]
        x = torch.cat(tensor_list, dim=-1)

        stats = self.net(x)
        mean = stats[...,:self.zdim]
        lvar = stats[...,self.zdim:]

        stdev = torch.exp(0.5*lvar)
        return mean, stdev 

    def forward(self, *args):
        mu, sig = self.get_stats(*args)
        return dists.normal.Normal(mu, sig)

class GaussianRNN(nn.Module):
    def __init__(self, xdim, hdim, zdim):
        super().__init__()
        self.zdim = zdim 
        self.rnn = nn.GRU(xdim, hdim, batch_first=True)
        self.net = GaussianNet(hdim, hdim, zdim)

    def forward(self, *args):
        tensor_list = [i for i in args if i is not None]
        x = torch.cat(tensor_list, dim=-1)
        hs, _ = self.rnn(x)
        
        z_dist = self.net(hs)
        return z_dist
