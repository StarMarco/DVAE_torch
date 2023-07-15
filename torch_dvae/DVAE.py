import torch 
import torch.nn as nn 
import torch.distributions as dists

from torch_dvae.transition_models.base import TransitionBase
from torch_dvae.measurement_models.base import MeasureBase
from torch_dvae.encoder_models.base import EncoderBase
from torch_dvae.inference_models.base import InferenceBase
from torch_dvae.initializers.base import InitializeBase

class DVAE(nn.Module):
    def __init__(self, inference: InferenceBase, 
                        transition: TransitionBase, 
                        measure: MeasureBase, 
                        y_encoder: EncoderBase, 
                        x_encoder: EncoderBase, 
                        init_net: InitializeBase):
        """
        Dynamical Variational Autoencoder (DVAE) class.
        This class takes the various components that make up a DVAE as inputs. 
        Custom classes of each of the components can be made using the base metaclasses as a template class 
        e.g. a custom encoder could be made using the EncoderBase metaclass. 

        For the encoders there is an EmptyEncoder class which can be used if one does not need to encode a certain 
        variable. 
        e.g. if a conditional variational autoencoder is not required set x_encoder = EmptyEncoder and whenever 
        the conditional variables "x" are to be encoded the x_encoder will not return anything and the DVAE will function 
        as a non-conditional DVAE. 

        *Note if more details are needed on the input models for the DVAE class please see the Base metaclasses for the 
        corresponding model. 

        **Note 2. Most of the model classes have a property for when they are initialized if a dimension is set to zero 
        then that variable is ignored as an input. 
        e.g. the Transition class may be initialized with (zdim, hdim, ydim, xdim). Since z, y and x are the latent,
        input and conditional variables respectively if ydim=0, then when "y" is inputted into the transition model during DVAE 
        evaluation or training, it is effectively ignored by the transition model. Hence, instead of the transition model 
        representing p(z_t| z_{1:t-1}, y_{1:t-1}, x_{1:T}) it becomes p(z_t| z_{1:t-1}, x_{1:T}) as y_{1:t-1} are ignored as inputs 

        Inputs:
            inference (InferenceBase): The inference model created using the InferenceBase metaclass.
                This model is used to calculate the inference latent variables which are used to train the DVAE 
                but is no longer needed when testing and evaluating the model. 
                Model: p(z_t| z_{1:t-1}, y_{1:T}, x_{1:T})
            transition (TransitionBase): The transition model created using the TransitionBase metaclass. 
                This model takes the previous variables (latent, input and conditional or a combination of these)
                and calculates the current time latent variable. ie. it is the dynamic model that determines the 
                latent variable dynamics through time. 
                Model: p(z_t| z_{1:t-1}, y_{1:t-1}, x_{1:T})
            measure (MeasureBase): The measurement model created using the MeasureBase metaclass.
                Takes the current time latent variable, previous measurements/inputs and the sequence of conditional variables 
                (or a combination of these) and returns the current input estimate
                Model: p(y_t| z_{1:t}, y_{1:t-1}, x_{1:T})
            y_encoder (EncoderBase): The model that encodes sequences of the input variables (y) using the EncoderBase metaclass.
                Encodes a sequence of variables y_{1:T} into a format the DVAE can use e.g. a RNN would do this by taking y_{1:T} 
                and converting it into a hidden variable h_T that "encodes" the sequential information into that variable. 
                Note, these have forward and backward methods that encode y_{1:t} up to a certain time "t" and the backward method 
                can then encode a noncausal sequence y_{1:T}. There is also a real-time encode method if we are encoding y_{1:T} 
                one variable at a time in a "for-loop" instead of using the entire sequence as an input up front. 
            x_encoder (EncoderBase): The model that encodes the conditional variable (x) sequences using the EncoderBase metaclass.
                Same idea as the y_encoder but this works on the conditional variable sequences x_{1:T}.
            init_net (InitializeBase): This initializes the latent variable (z_0) using the InitializeBase metaclass. 
                Note the sequences start with "1" e.g. x_{1:T}, y_{1:T}, z_{1:T}. Hence, this variable is like an RNNs initial 
                hidden variable; it does not end up being returned by the model but is simply used to start the calculation of 
                other variables. 
        """
        super().__init__()
        self.xdim = x_encoder.xdim 
        self.zdim = transition.zdim 

        self.transition = transition 
        self.measure = measure 
        self.x_encoder = x_encoder 
        self.y_encoder = y_encoder
        self.inference = inference 
        self.init_net = init_net

    def tile(self, tensor, shape, dim):
        """
        Helper method that repeats a tensor across a dimension (dim).
        Number of repeats is determined by the "shape" variable 
        """
        if tensor == None:
            return None 
        else:
            out = torch.tile(tensor.unsqueeze(dim), shape)
        return out
    
    def inference_func(self, xs, ys):
        """
        given the conditional variable sequences (xs) and the input sequences (ys)
        returns the latent inference variable sequences. 

        The most general inference model can be expressed as:
            p(z_t| z_{1:t-1}, y_{1:T}, x_{1:T})
        
        Inputs: 
            xs (tensor): conditional variable sequences, size (bs, seq, xdim)
            ys (tensor): input variable sequences, size (bs, seq, ydim)

        Outputs: 
            z_inf_dists (torch.distribution): a torch distribution method describing the 
                latent variables. If you were to sample from it once the size should be,
                (bs, seq, zdim)
            y_1t (tensor): the variables output from the y_encoder's forward encoding (causal encoding)
                size, (bs, seq, yenc_dim)
            x_1t (tensor): the variables output from the x_encoder's forward encoding 
                size, (bs, seq, xenc_dim)
            y_1T (tensor): the variables output from a forward and backward encoding 
                representing a noncausal sequence, size (bs, seq, yenc_dim)
            x_1T (tensor): the variables output from a forward and backward encoding 
                representing a noncausal sequence, size (bs, seq, yenc_dim)
        """
        y_1t = self.y_encoder.encode_fwd(ys)
        y_1T = self.y_encoder.encode_bwd(ys, y_1t)

        x_1t = self.x_encoder.encode_fwd(xs)
        x_1T = self.x_encoder.encode_bwd(xs, x_1t)

        zs_inf_dists = self.inference.get_dist(y_1T, x_1T)
        return zs_inf_dists, y_1t, x_1t, y_1T, x_1T

    def get_priors(self, z0, zs, ys_enc, xs_enc):
        """
        Given an initial latent variable, an inference latent sequence and encoded 
        x and y sequences this returns the transitioned latent variable sequences 
        i.e. takes z_{0,T-1} inference sequence and transitions each variable (in parallel, 
        z0_inf -> z1, z1_inf -> z2) to find sequence z_{1:T}.

        Inputs: 
            z0 (tensor): the initial latent dummy variable, size (bs, zdim)
            zs (tensor): the inference latent sequence, size (bs, seq, zdim)
            ys_enc (tensor): encoded causal sequence (y_1:t), size (bs, seq, yenc_dim)
            xs_enc (tensor): encoded noncausal sequence (x_1:T), size (bs, seq, xenc_dim)

        Outputs:
            zs_pri_dists (torch.distribution): The prior latent distribution from the 
                transition class which is generally expressed as, 
                    p(z_t| z_{1:t-1}, y_{1:t-1}, x_{1:T}) 
                if sampled from the samples should be size (bs, seq, zdim)
        """
        z_tm1 = torch.cat([z0.unsqueeze(1), zs[:,:-1,:]], dim=1)

        zs_pri_dists = self.transition.get_dist(z_tm1, ys_enc, xs_enc)
        return zs_pri_dists

    def get_loss(self, xs, ys):
        """
        Using the input and conditional variables finds the loss to train the DVAE 
        i.e. the negative log-likelihood and the KL-divergence. 

        Inputs: 
            xs (tensor): conditional variables, size (bs, seq, xdim)
            ys (tensor): input variables, size (bs, seq, ydim)

        Outputs: 
            nll (tensor): the negative log-likelihood (nll) found using the measurement model,
                which can generally be expressed as p(y_t| y_{1:t-1}, z_{1:t}, x_{1:T}).
                using the log_prob method with the measurement model and the observations (ys)
                one can find the nll of size. 
                Note it is summed across the sequence dimension as the sum of log_probs is the 
                same is multiplying the non-log probabilities across time.
                i.e. p(y_1:T) = p(y_1)p(y2) ... p(y_T) => log p(y_1:T) = log p(y1) + log p(y2) ... + log p(yT)

            kl (tensor): the KL-divergence between the inference latent distributions and the prior/transition 
                distributions. 

            *Note both of these are scalars (size 1) as they are loss terms. 
        """
        # --- Inference --- 
        zs_inf_dists, y_1t, x_1t, y_1T, x_1T = self.inference_func(xs, ys)
        zs = zs_inf_dists.sample()  

        # --- Transition func --- 
        z0_dist = self.init_net(ys, xs)
        z0 = z0_dist.sample()
        zs_pri_dists = self.get_priors(z0, zs, y_1t, x_1T)

        # --- Measure func --- 
        y_dists = self.measure.get_dist(zs, y_1t, x_1T)
        
        # --- Losses --- 
        nll = -y_dists.log_prob(ys).sum(1).mean()   
        kl =  dists.kl.kl_divergence(zs_inf_dists, zs_pri_dists).sum(1).mean()
        return nll, kl 

    def get_stats(self, dist):
        """
        Since we use Gaussians here this is a helper method that returns 
        the mean and standard deviation (although Gaussians are not strictly 
        necessary and in the future other distributions could be exprimented with)
        """
        mean = dist.loc
        std = dist.scale 
        return mean, std

    def noncausal_forward(self, xs, N=1):
        """
        If control inputs are noncausal i.e. x_{1:T} then use this method when 
        generating latent variables and observations using the transition and measurement models

        i.e. z_t ~ p(z_t| z_{1:t-1}, y_{1:t-1}, x_{1:T})
             y_t ~ p(y_t| y_{1:t-1}, z_{1:t}, x_{1:T})

        use these 2 models recursively until z_{1:T} and y_{1:T} is found. Note the above 
        distributions can be sampled "N" times to find N z_{1:T} and y_{1:T} sequences. 

        Inputs:
            xs (tensor): controls sequence, size (bs, seq, xdim)
            N (int): number of samples from the latent space 
        
        Outputs: 
            z_dists (torch.dist): normal distribution describing the generated latent sequence 
            y_dists (torch.dist): normal distribution describing the generated observation sequence 
            zs (tensor): raw latent samples generated, size (N, bs, seq, zdim)
            ys (tensor): raw output samples generated, size (N, bs, seq, ydim)
        """
        # set up variables 
        bs, seq = xs.shape[:2]
        hdim = self.y_encoder.hdim
        h = torch.zeros([bs, hdim]).to(xs.device)

        x_1t = self.x_encoder.encode_fwd(xs)
        x_1T = self.x_encoder.encode_bwd(xs, x_1t)

        z0_dist = self.init_net(xs, xs)
        z = z0_dist.sample([N])

        # storage 
        z_mean_store = []
        z_stds_store = [] 
        y_mean_store = [] 
        y_stds_store = [] 
        zs_store = [] 
        ys_store = [] 
        for i in range(seq):
            # --- get control var. ---
            x = x_1T[:,i,:]      
            x = self.tile(x, (N,1,1), dim=0)

            # --- transition latent ---
            h_tile = self.tile(h, (N,1,1), dim=0)
            z_dist = self.transition.get_dist(z, h_tile, x)  # (N, bs, zdim)
            z = z_dist.sample()  

            # --- convert to measurement space ---
            y_dist = self.measure.get_dist(z, h_tile, x)

            y = y_dist.sample()  

            # Note if ydim is set to 0 the measurement and transition models can ignore this variable 
            # (which is a bit inefficent since this would be computed even if it is never used)
            h = self.y_encoder.encode(y[0,:,:], h)

            # --- get stats and store ---
            y_mean, y_stds = self.get_stats(y_dist)
            z_mean, z_stds = self.get_stats(z_dist)
            
            z_mean_store.append(z_mean[0])  # get 1st sample's outputs
            z_stds_store.append(z_stds[0])
            y_mean_store.append(y_mean[0])
            y_stds_store.append(y_stds[0])
            zs_store.append(z)
            ys_store.append(y)
        
        z_mean = torch.stack(z_mean_store, dim=1)
        z_stds = torch.stack(z_stds_store, dim=1)
        y_mean = torch.stack(y_mean_store, dim=1)
        y_stds = torch.stack(y_stds_store, dim=1)
        
        z_dists = dists.normal.Normal(z_mean, z_stds)
        y_dists = dists.normal.Normal(y_mean, y_stds)
        zs = torch.stack(zs_store, dim=-2)
        ys = torch.stack(ys_store, dim=-2)

        return z_dists, y_dists, zs, ys

    def reconstruct(self, ys, xs):
        """
        Used in the unsupervised DVAE case (hence xs and ys are really both
        input variables ys). This method is used to test how well the DVAE 
        can reconstruct the input sequences. 
        """
        zs_inf_dists, y_1t, x_1t, y_1T, x_1T = self.inference_func(xs, ys)
        zs = zs_inf_dists.sample()  
        y_dists = self.measure.get_dist(zs, y_1t, x_1T)
        return zs_inf_dists, y_dists 