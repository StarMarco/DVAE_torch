import argparse
import json
import time 
import os 

import torch 
import torch.nn as nn
import torch.distributions as dists 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb 

from data.dataprep import DataPrep
from data.utils import win_to_seq, sliding_window
from torch_dvae.measurement_models import * 
from torch_dvae.transition_models import * 
from torch_dvae.encoder_models import * 
from torch_dvae.inference_models import * 
from torch_dvae.initializers import * 
from torch_dvae.DVAE import DVAE 

from torch_dvae.utils import score_func
from training import RunningAverage, transition_selector, measurement_selector, inference_selector, \
                    init_selector, encoder_selector, select_dims, str2bool

sb.set_theme()  # super important (do not delete this)

def make_model(xdim, hdim, zdim, ydim, transition, measurement, inference, initializer, encoder, 
               transition_inp, measurement_inp, init_inp, has_controls):
    if has_controls:
        x_encoder = encoder_selector(encoder, xdim, hdim)
    else: 
        x_encoder = encoder_selector("none", xdim, hdim)   # empty encoder returns None for all encodings of "xs"
        xdim = 0 
    y_encoder = encoder_selector(encoder, ydim, hdim)

    t_zdim, t_ydim, t_xdim = select_dims(transition_inp, zdim, y_encoder.hdim, x_encoder.hdim)
    m_zdim, m_ydim, m_xdim = select_dims(measurement_inp, zdim, y_encoder.hdim, x_encoder.hdim)
    i_zdim, i_ydim, i_xdim = select_dims(init_inp, zdim, ydim, xdim)


    transition_model = transition_selector(transition, t_zdim, hdim, t_ydim, t_xdim)
    measurement_model = measurement_selector(measurement, ydim, m_zdim, hdim, m_ydim, m_xdim)
    inference_model = inference_selector(inference, zdim, hdim, y_encoder.hdim, x_encoder.hdim)  # force user to have y_{1:T} and x_{1:T} as inputs for now 

    initializer_model = init_selector(initializer, hdim, zdim, i_ydim, i_xdim)

    model = DVAE(inference_model, transition_model, measurement_model, y_encoder, x_encoder, initializer_model)

    return model 

def extract_input_target_pairs(xs, ys):
    """
    Extracts the labelled pairs from the semi-supervised dataset 
    (-1 is placed in the target tensor, ys, to indicate an unlabelled datapoint)
    """
    y_shape = (-1,) + tuple(ys.shape[1:])
    x_shape = (-1,) + tuple(xs.shape[1:])

    mask = (ys != -1)[:,:,0]
    lb_ys = ys[mask].reshape(y_shape)
    lb_xs = xs[mask].reshape(x_shape)
    return lb_xs, lb_ys 

class SemiSupervisedTrainer:
    def __init__(self, lr, L2):
        self.lr = lr 
        self.L2 = L2 

    def unsupervised_forward(self, model: DVAE, xs):
        # --- Inference --- 
        zs_inf_dists, y_1t, x_1t, y_1T, x_1T = model.inference_func(xs, xs)
        zs = zs_inf_dists.sample()  

        # --- Transition func --- 
        z0_dist = model.init_net(xs, xs)
        z0 = z0_dist.sample()
        zs_pri_dists = model.get_priors(z0, zs, y_1t, x_1T)

        # --- Measure func --- 
        x_dists = model.measure.get_dist(zs, y_1t, x_1T)
        
        # --- Losses --- 
        nll = -x_dists.log_prob(xs).sum(1).mean()   
        kl =  dists.kl.kl_divergence(zs_inf_dists, zs_pri_dists).sum(1).mean()
        return nll, kl, x_dists, zs_inf_dists

    def train_step(self, unsupervised_model: DVAE, model: DVAE, xs, ys, optimizer):
        # --- unsupervised stage --- 
        nll, kl, xs_dists, zs_dists = self.unsupervised_forward(unsupervised_model, xs)
        unsupervised_loss = nll + kl 

        # --- process data for supervised stage --- 
        zs = zs_dists.sample()        
        xs_rec = xs_dists.sample()     
        
        xs = torch.cat([xs_rec, zs], dim=-1)    # new inputs are latent and sensor values combined 

        # --- supervised stage --- 
        lb_xs, lb_ys = extract_input_target_pairs(xs, ys)
        if torch.numel(lb_xs) == 0:
            supervised_loss = torch.zeros([1]).to(xs.device)     # setting supervised_loss = 0 if a batch has no labelled data 
                                            # this ensures the network parameters don't change too much given no supervised data  
        else:
            nll, kl = model.get_loss(lb_xs, lb_ys)
            supervised_loss = nll + kl 
        
        loss = unsupervised_loss + supervised_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return unsupervised_loss, supervised_loss
    
    def valid_step(self, unsupervised_model: DVAE, model: DVAE, xs, ys):
        # --- unsupervised stage --- 
        nll, kl, xs_dists, zs_dists = self.unsupervised_forward(unsupervised_model, xs)
        unsupervised_loss = nll + kl 

        # --- process data for supervised stage --- 
        zs = zs_dists.loc        
        xs_rec = xs_dists.loc     
        
        xs = torch.cat([xs_rec, zs], dim=-1)

        # --- supervised stage --- 
        lb_xs, lb_ys = extract_input_target_pairs(xs, ys)
        if torch.numel(lb_xs) == 0:
            supervised_loss = torch.zeros([1]).to(xs.device)     # setting supervised_loss = 0 if a batch has no labelled data 
                                            # this ensures the network parameters don't change too much given no supervised data  
        else:
            nll, kl = model.get_loss(lb_xs, lb_ys)
            supervised_loss = nll + kl 
        
        loss = unsupervised_loss + supervised_loss

        return loss
    
    def train_model(self, epochs, train_loader, valid_loader, semi_supervised_model, 
                    model_PATH, device):
        best_loss = 1e10
        logger = RunningAverage()
        logger.add_key(["unsupervised loss", "supervised loss", "valid loss"])
        optimizer = torch.optim.Adam(semi_supervised_model.parameters(), self.lr, weight_decay=self.L2)

        unsupervised_model = semi_supervised_model.unsupervised_model
        model = semi_supervised_model.model

        for epoch in range(1, epochs+1):
            logger.reset_all()  # reset average counter and losses to zero 

            # --- Training --- 
            for xs, ys in train_loader:
                xs = xs.to(device).float()
                ys = ys.to(device).float()

                unsuper_loss, super_loss = self.train_step(unsupervised_model, model, xs, ys, optimizer)
                logger.add_loss(super_loss, "supervised loss")
                logger.add_loss(unsuper_loss, "unsupervised loss")

            # --- Validation --- 
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                with torch.no_grad():
                    for xs, ys in valid_loader:
                        xs = xs.to(device).float()
                        ys = ys.to(device).float()

                        valid_loss = self.valid_step(unsupervised_model, model, xs, ys)
                        logger.add_loss(valid_loss, "valid loss")

                    # average losses 
                    logger.avg_loss()   

                    # store losses 
                    super_loss = logger.get_avg_loss("supervised loss")
                    unsuper_loss = logger.get_avg_loss("unsupervised loss")
                    valid_loss = logger.get_avg_loss("valid loss")

                    if valid_loss < best_loss:
                        best_loss = valid_loss 
                        torch.save(semi_supervised_model.state_dict(), model_PATH)
                        message = "new best loss, saving model ..."
                    else:
                        message = ""

                    print(("Epoch {}/{}, unsupervised loss: {:.4f}, supervised loss: {:.4f}, valid loss: {:.4f} " + message)
                    .format(epoch, epochs, unsuper_loss, super_loss, valid_loss)) 

        semi_supervised_model.load_state_dict(torch.load(model_PATH))   # load the best performing model 
        return semi_supervised_model

class SemiSupervisedModel(nn.Module):
    def __init__(self, unsupervised_model, model):
        super().__init__()
        self.unsupervised_model = unsupervised_model
        self.model = model 


def test_semisupervised_model(semi_supervised_model, test_x, test_t, test_y, T, N, device):
    results = {
        "y_true": [],
        "RMSE": 0, 
        "y_nll": 0, 
        "score": 0, 
        "y_mean": [],
        "y_stds": [], 
        "z_mean": [],
        "z_stds": [],
        "zs": [],
        "ys": [],
        "x_true": [],
        "x_mean": [],
        "x_stds": [], 
        "times": [] 
    }
    model = semi_supervised_model.model
    unsupervised_model = semi_supervised_model.unsupervised_model
    with torch.no_grad():
        MSE = [] 
        NLL = [] 
        scores = [] 
        T = int(T)
        for i, x in enumerate(test_x):
            y = test_y[i][0,:,:].to(device).float()
            x = x[0,:,:].to(device).float()
            t = test_t[i][0,:,0].to(device).float()

            # --- get time windowed data --- 
            x = sliding_window(x, T)
            y = sliding_window(y, T)

            # --- generate inputs and latent variables ---
            z_dist, x_dist = unsupervised_model.reconstruct(x, x)
            zs = z_dist.loc         
            x_mean = x_dist.loc     
            x_stds = x_dist.scale   

            x_new = torch.cat([x, zs], dim=-1) # reconstructed sensors and latent 

            _, _, zs, ys = model.noncausal_forward(x_new, N)

            z_dist = dists.normal.Normal(zs.mean(0), zs.std(0))
            y_dist = dists.normal.Normal(ys.mean(0), ys.std(0))

            z_mean = z_dist.loc     
            z_stds = z_dist.scale   
            y_mean = y_dist.loc     
            y_stds = y_dist.scale   

            # --- convert back to seq ---
            x = win_to_seq(x)
            x_mean = win_to_seq(x_mean)
            x_stds = win_to_seq(x_stds)
            y = win_to_seq(y)
            z_mean = win_to_seq(z_mean)
            z_stds = win_to_seq(z_stds)
            y_mean = win_to_seq(y_mean)
            y_stds = win_to_seq(y_stds)
            zs = win_to_seq(zs)
            ys = win_to_seq(ys)

            y_dist = dists.normal.Normal(y_mean, y_stds)
            # --- get losses --- 
            nll = -y_dist.log_prob(y).sum(0).mean()
            mse = (y_mean - y) ** 2
            score = score_func(y_mean[-1,:], y[-1])

            # --- store variables --- 
            MSE.append(mse.detach().cpu().numpy())
            NLL.append(nll.detach().cpu().numpy())
            scores.append(score.detach().cpu().numpy())
            results["y_true"].append(y.detach().cpu().numpy())
            results["y_mean"].append(y_mean.detach().cpu().numpy())
            results["y_stds"].append(y_stds.detach().cpu().numpy())
            results["z_mean"].append(z_mean.detach().cpu().numpy())
            results["z_stds"].append(z_stds.detach().cpu().numpy())
            results["zs"].append(zs.detach().cpu().numpy())
            results["ys"].append(ys.detach().cpu().numpy())
            results["x_true"].append(x.detach().cpu().numpy())
            results["x_mean"].append(x_mean.detach().cpu().numpy())
            results["x_stds"].append(x_stds.detach().cpu().numpy())
            results["times"].append(t.detach().cpu().numpy())

    MSE = np.concatenate(MSE, axis=0)
    RMSE = np.sqrt(MSE.mean())
    results["RMSE"] = RMSE 

    nll = sum(NLL) / len(NLL)   # mean nll over all units 
    results["y_nll"] = nll 

    scores = np.concatenate(scores, axis=0).sum()
    results["score"] = scores 

    return results 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--save_path", type=str, default="saved_models/DVAE")
    parser.add_argument("--zdim", type=int, default=2)
    parser.add_argument("--hdim", type=int, default=50)
    parser.add_argument("--T", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L2", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=250)
    parser.add_argument("--split", type=float, default=90)
    parser.add_argument("--valid_split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--transition", type=str, default="mlp")
    parser.add_argument("--measurement", type=str, default="mlp")
    parser.add_argument("--inference", type=str, default="rnn")
    parser.add_argument("--encoder", type=str, default="rnn")
    parser.add_argument("--initializer", type=str, default="controls")
    parser.add_argument("--transition_inputs", type=list, default="zx")
    parser.add_argument("--measurement_inputs", type=str, default="zx")
    parser.add_argument("--init_inputs", type=str, default="yx")
    # All the pre- arguements are for choosing the unsupervised trained DVAE model 
    parser.add_argument("--pre_transition", type=str, default="mlp")
    parser.add_argument("--pre_measurement", type=str, default="mlp")
    parser.add_argument("--pre_inference", type=str, default="rnn")
    parser.add_argument("--pre_encoder", type=str, default="rnn")
    parser.add_argument("--pre_initializer", type=str, default="measurement")
    parser.add_argument("--pre_transition_inputs", type=list, default="zy")
    parser.add_argument("--pre_measurement_inputs", type=str, default="zy")
    parser.add_argument("--pre_init_inputs", type=str, default="yx")
    parser.add_argument("--controls", type=str2bool, default=False, const=True, nargs="?")
    args = parser.parse_args()

    PATH = "CMAPSS"
    prep_class = DataPrep(PATH, args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, x_test, y_test, t_test = prep_class.semi_supervised_dataprep(args.T, args.bs, args.split, args.valid_split)

    # --- Load Unsupervised Model --- 
    if not args.controls:
        args.pre_initializer = "measure"    # can't use control variables if they don't exist; use measurements instead 

    xdim = x_test[0].shape[-1]
    hdim = args.hdim
    zdim = args.zdim
    T = args.T

    unsupervised_model = make_model(xdim, hdim, zdim, xdim, args.pre_transition, args.pre_measurement, args.pre_inference, args.pre_initializer,
                        args.pre_encoder, args.pre_transition_inputs, args.pre_measurement_inputs, args.pre_init_inputs, args.controls)

    unsupervised_model = unsupervised_model.to(device)

    # --- Train RUL DVAE with Preprocessed Data --- 
    ydim = 1 

    supervised_model = make_model(xdim+zdim, args.hdim, args.zdim, ydim, args.transition, args.measurement,
                        args.inference, args.initializer, args.encoder, args.transition_inputs,
                        args.measurement_inputs, args.init_inputs, True)
    supervised_model = supervised_model.to(device)

    model = SemiSupervisedModel(unsupervised_model, supervised_model)

    trainer = SemiSupervisedTrainer(args.lr, args.L2)

    save_PATH = args.save_path + "_" + args.transition + "_" \
                + args.measurement + "_" + args.inference + "_" \
                + args.encoder + "_" + args.initializer + "_" \
                + args.dataset + "_semi" + "_" + str(args.split) + ".pth"
    
    begin = time.time()
    trainer.train_model(args.epochs, train_loader, valid_loader, model, save_PATH, device)
    end = time.time()

    runtime = end - begin 
    print(f"Training runtime: {runtime/60} minutes")

    model_params = {
        "xdim": xdim, 
        "hdim": args.hdim, 
        "zdim": args.zdim,
        "ydim": ydim,
        "T": args.T, 
        "transition inputs": args.transition_inputs,
        "measurement inputs": args.measurement_inputs,
        "initializer inputs": args.init_inputs,
        "transition": args.transition,
        "measurement": args.measurement,
        "inference": args.inference,
        "initializer": args.initializer,
        "encoder": args.encoder,
        "unsupervised transition inputs": args.pre_transition_inputs, 
        "unsupervised measurement inputs": args.pre_measurement_inputs, 
        "unsupervised initializer inputs": args.pre_init_inputs,
        "unsupervised transition": args.pre_transition,
        "unsupervised measurement": args.pre_measurement,
        "unsupervised inference": args.pre_inference,
        "unsupervised initializer": args.pre_initializer,
        "unsupervised encoder": args.pre_encoder,
        "bs": args.bs,
        "lr": args.lr,
        "L2": args.L2,
        "percentage cut from dataset": args.split,
        "valid split": args.valid_split,
        "train time": runtime,
        "epochs": args.epochs
    }

    json_save = save_PATH[:-4] + ".json"    # get rid of .pth extension and add .json 
    print("saving model construction hyperparameters in " + json_save)
    with open(json_save, "w") as outfile:
        json.dump(model_params, outfile)