import argparse
import json
import time 

import torch 
import torch.distributions as dists 
import numpy as np 

from data.dataprep import DataPrep
from torch_dvae.measurement_models import * 
from torch_dvae.transition_models import * 
from torch_dvae.encoder_models import * 
from torch_dvae.inference_models import * 
from torch_dvae.initializers import * 
from torch_dvae.DVAE import DVAE 

# ------------------------------------
#           Helpful Utils
# ------------------------------------

def str2bool(v):
    """
    Used for boolean arguments in argparse; avoiding `store_true` and `store_false`.
    """
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class RunningAverage():
    def __init__(self):
        self.running_losses = {}
        self.names = [] 

    def add_key(self, names):
        # names are a list of strings for the dictionary keys containing losses
        for name in names:
            self.running_losses[name] = []
            self.running_losses[name + "_avg"] = 0 
            self.running_losses[name + "_std"] = 0
            
            self.names.append(name)

    def add_loss(self, loss, name):
        self.running_losses[name].append(loss.item())
        
    def avg_loss(self):
        for name in self.names:
            avg = np.array(self.running_losses[name]).mean()
            self.running_losses[name + "_avg"] = avg 

    def std_loss(self):
        for name in self.names:
            std = np.array(self.running_losses[name]).std()
            self.running_losses[name + "_std"] = std

    def reset_all(self):
        # reset all losses and counters to 0 
        for name in self.running_losses:
            self.running_losses[name] = [] 

    def get_avg_loss(self, name):
        return self.running_losses[name + "_avg"]

    def get_std_loss(self, name):
        return self.running_losses[name + "_std"]

# --- Data prep ---
def prep_data(prep_class, T, bs, split, stride=1, unsupervised=False):
    x_train, y_train, t_train = prep_class.prep_data(prep_class.ntrain, T, stride)
    x_train, y_train, t_train, x_valid, y_valid, t_valid = prep_class.valid_set(x_train, y_train, t_train, split)
    x_test, y_test, t_test = prep_class.prep_test(prep_class.ntest, prep_class.RUL)

    if unsupervised:
        train_loader, valid_loader = prep_class.get_dataloaders(bs, x_train, x_train, x_valid, x_valid)
    else:
        train_loader, valid_loader = prep_class.get_dataloaders(bs, x_train, y_train, x_valid, y_valid)
    return train_loader, valid_loader, x_test, y_test, t_test

def get_dataclass(PATH, dataset):
    prep_class = DataPrep(PATH, dataset)

    if dataset == "FD001" or dataset == "FD003":
        prep_class.op_normalize(K=1)    # K=1 normalization, K=6 operating condition norm 
    else: 
        prep_class.op_normalize(K=6)
    return prep_class 

# --- Selectors ---
def transition_selector(name: str, zdim: int, hdim: int, ydim, xdim):
    transition = MLPTransition(zdim, hdim, ydim, xdim)
    if name == "mlp":
        transition = MLPTransition(zdim, hdim, ydim, xdim)
    else:
        print(name + ": transition model name not valid, selecting MLP transition model as default")
    return transition

def measurement_selector(name: str, ydim: int, zdim: int, hdim: int, yenc: int, xdim: int):
    measure = MLPMeasure(ydim, zdim, hdim, yenc, xdim)
    if name == "mlp":
        measure = MLPMeasure(ydim, zdim, hdim, yenc, xdim)
    else: 
        print(name + ": measurement model name not valid, selecting MLP measurement model as default")
    return measure 

def encoder_selector(name: str, xdim: int, hdim: int):
    encoder = RNNEncoder(xdim, hdim)
    if name == "none":
        encoder = EmptyEncoder()
    elif name == "rnn":
        encoder = RNNEncoder(xdim, hdim)
    elif name == "init":
        encoder = InitEncoder(xdim, hdim)

    else:
        print(name + ": encoder name not valid, selecting Reccurent encoder model as default")
    return encoder 

def inference_selector(name: str, zdim: int, hdim: int, ydim: int, xdim: int):
    inference = RNNInference(zdim, hdim, ydim, xdim)
    if name == "rnn":
        inference = RNNInference(zdim, hdim, ydim, xdim)
    else:
        print(name + ": inference model name not valid, selecting Reccurent inference model as default")
    return inference 

def init_selector(name: str, hdim: int, zdim: int, ydim: int, xdim: int):
    init_net = ControlInitializer(hdim, zdim, ydim, xdim)
    if name == "controls":
        init_net = ControlInitializer(hdim, zdim, ydim, xdim)
    elif name == "measure":
        init_net = MeasureInitializer(hdim, zdim, ydim, xdim)
    else: 
        print(name + ": initializer name not valid, selecting Control based initializer as default")
    return init_net 

def select_dims(input_str: str, zenc: int, yenc: int, xenc: int):
    zdim = 0 
    ydim = 0 
    xdim = 0 
    for inp in input_str:
        if inp == "z":
            zdim = zenc
        elif inp == "x":
            xdim = xenc 
        elif inp == "y":
            ydim = yenc 
    return zdim, ydim, xdim 

# ------------------------------------
#           Training Class 
# ------------------------------------
class Trainer(): 
    def __init__(self, lr, L2):
        self.lr = lr 
        self.L2 = L2 

    def train_step(self, model: DVAE, xs, ys, optimizer):
        nll, kl = model.get_loss(xs, ys)
        loss = nll + kl 
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return nll, kl 

    def valid_step(self, model: DVAE, xs, ys, unsupervised):
        if unsupervised:
            _, y_dists = model.reconstruct(ys, xs)
        else:
            _, y_dists, _, _ = model.noncausal_forward(xs)
            
        nll = -y_dists.log_prob(ys).sum(1).mean()   
        mse = ((y_dists.loc - ys) ** 2).mean() 
        return nll, mse

    def train_model(self, epochs, train_loader, valid_loader, model, model_PATH, device, unsupervised=False, print_loss=True):
        best_loss = 1e10
        logger = RunningAverage()
        logger.add_key(["train nll", "valid nll", "train kl", "valid mse"])
        optimizer = torch.optim.Adam(model.parameters(), self.lr, weight_decay=self.L2)

        for epoch in range(1, epochs+1):
            logger.reset_all()  # reset average counter and losses to zero 

            # --- Training --- 
            for xs, ys in train_loader:
                xs = xs.to(device).float()
                ys = ys.to(device).float()

                nll, kl = self.train_step(model, xs, ys, optimizer)
                logger.add_loss(nll, "train nll")
                logger.add_loss(kl, "train kl")

            # --- Validation --- 
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                with torch.no_grad():
                    for xs, ys in valid_loader:
                        xs = xs.to(device).float()
                        ys = ys.to(device).float()

                        valid_nll, valid_mse = self.valid_step(model, xs, ys, unsupervised)
                        logger.add_loss(valid_nll, "valid nll")
                        logger.add_loss(valid_mse, "valid mse")

                    # average losses 
                    logger.avg_loss()   

                    # store losses 
                    train_nll = logger.get_avg_loss("train nll")
                    train_kl = logger.get_avg_loss("train kl")
                    valid_nll = logger.get_avg_loss("valid nll")
                    valid_rmse = np.sqrt(logger.get_avg_loss("valid mse"))

                    if valid_nll < best_loss:
                        best_loss = valid_nll 
                        torch.save(model.state_dict(), model_PATH)
                        message = "new best loss, saving model ..."
                    else:
                        message = ""

                    if print_loss:
                        print(("Epoch {}/{}, kl-div: {:.4f}, nll: {:.4f}, valid nll: {:.4f}, valid RMSE: {:.4f} " + message)
                        .format(epoch, epochs, train_kl, train_nll, valid_nll, valid_rmse))

        model.load_state_dict(torch.load(model_PATH))   # load the best performing model 
        return model

    def validate(self, valid_loader, model, device, loop=10):
        """
        Used as a seperate validation method for hyperparameter optimization 
        """
        logger = RunningAverage()
        logger.add_key(["valid nll", "valid mse"])
        logger_total = RunningAverage()
        logger_total.add_key(["valid nll", "valid mse"])
        with torch.no_grad():
            for i in range(loop):
                for xs, ys in valid_loader:
                    xs = xs.to(device).float()
                    ys = ys.to(device).float()

                    valid_nll, valid_mse = self.valid_step(model, xs, ys, False)
                    logger.add_loss(valid_nll, "valid nll")
                    logger.add_loss(valid_mse, "valid mse")

                # average losses 
                logger.avg_loss()   

                # store losses 
                valid_nll = logger.get_avg_loss("valid nll")
                valid_mse = logger.get_avg_loss("valid mse")
                logger_total.add_loss(valid_nll, "valid nll")
                logger_total.add_loss(valid_mse, "valid mse")

            logger_total.avg_loss()
            logger_total.std_loss()
            total_nll = logger_total.get_avg_loss("valid nll")
            total_mse = logger_total.get_avg_loss("valid mse")
            total_nll_std = logger_total.get_std_loss("valid nll")
            total_mse_std = logger_total.get_std_loss("valid mse")


        return total_nll, total_nll_std, total_mse, total_mse_std

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
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--transition", type=str, default="mlp")
    parser.add_argument("--measurement", type=str, default="mlp")
    parser.add_argument("--inference", type=str, default="rnn")
    parser.add_argument("--encoder", type=str, default="rnn")
    parser.add_argument("--initializer", type=str, default="controls")
    parser.add_argument("--transition_inputs", type=str, default="zx")
    parser.add_argument("--measurement_inputs", type=str, default="zx")
    parser.add_argument("--init_inputs", type=str, default="yx")
    parser.add_argument("--controls", type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument("--unsupervised", type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument("--load_hyperparameters", type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument("--trial", type=int, default=12)
    args = parser.parse_args()

    from hyperparameters.utils import get_optimal_hyperparameters_args
    PATH = "CMAPSS"
    dataclass = get_dataclass(PATH, args.dataset)

    if args.load_hyperparameters:
        get_optimal_hyperparameters_args(dataclass, args)
        print("Loaded Optimized Hyperparameters: lr = {:.7f}, bs = {}, T = {}, stride = {}, L2 = {:.7f}, hdim = {}"
              .format(args.lr, args.bs, args.T, args.stride, args.L2, args.hdim))

    # --- Data prep --- 
    train_loader, valid_loader, x_test, y_test, t_test = prep_data(dataclass, stride=args.stride, T=args.T, bs=args.bs, split=args.split, unsupervised=args.unsupervised)

    # --- Create model and training instance ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xdim = x_test[0].shape[-1]

    if args.unsupervised:
        ydim = xdim     # sensors are the targets 
        path_append = "pre"
    else:
        ydim = 1        # RUL is 1D 
        path_append = ""
    
    if args.controls:
        x_encoder = encoder_selector(args.encoder, xdim, args.hdim)
    else: 
        x_encoder = encoder_selector("none", xdim, args.hdim)   # empty encoder returns None for all encodings of "xs"
        xdim = 0 
        if args.initializer == "controls":
            args.initializer = "measure"    # can't use control variables if they don't exist; use measurements instead 
    y_encoder = encoder_selector(args.encoder, ydim, args.hdim)
    
    t_zdim, t_ydim, t_xdim = select_dims(args.transition_inputs, args.zdim, y_encoder.hdim, x_encoder.hdim)
    m_zdim, m_ydim, m_xdim = select_dims(args.measurement_inputs, args.zdim, y_encoder.hdim, x_encoder.hdim)
    i_zdim, i_ydim, i_xdim = select_dims(args.init_inputs, args.zdim, ydim, xdim)


    transition_model = transition_selector(args.transition, t_zdim, args.hdim, t_ydim, t_xdim)
    measurement_model = measurement_selector(args.measurement, ydim, m_zdim, args.hdim, m_ydim, m_xdim)
    inference_model = inference_selector(args.inference, args.zdim, args.hdim, y_encoder.hdim, x_encoder.hdim)  # force user to have y_{1:T} and x_{1:T} as inputs for now 

    initializer = init_selector(args.initializer, args.hdim, args.zdim, i_ydim, i_xdim)
    assert x_encoder or y_encoder or transition_model or measurement_model or inference_model or initializer is not None, \
        "One of the model types specified in the arguements are not part of the model zoo. Please input another arguement."

    model = DVAE(inference_model, transition_model, measurement_model, y_encoder, x_encoder, initializer).to(device)   # type: ignore 

    # --- Training ---
    save_PATH = args.save_path + path_append + "_" + args.transition + "_" \
                + args.measurement + "_" + args.inference + "_" \
                + args.encoder + "_" + args.initializer + "_" \
                + args.dataset + ".pth"

    torch.save(model.state_dict(), save_PATH)

    trainer = Trainer(args.lr, args.L2)
    begin = time.time()

    model = trainer.train_model(args.epochs, train_loader, valid_loader, model, save_PATH, device, args.unsupervised)

    end = time.time()
    runtime = end - begin 
    print("Training runtime: {:.4f} minutes".format(runtime / 60.))

    # --- Store Hyperparameters and Model Info. --- 

    model_params = {
        "xdim": xdim, 
        "hdim": args.hdim, 
        "zdim": args.zdim,
        "ydim": ydim,
        "T": args.T, 
        "transition inputs": args.transition_inputs,
        "measurement inputs": args.measurement_inputs,
        "initializer inputs": args.init_inputs,
        "bs": args.bs,
        "lr": args.lr,
        "L2": args.L2,
        "valid split": args.split,
        "train time": runtime,
        "epochs": args.epochs,
    }

    json_save = save_PATH[:-4] + ".json"    # get rid of .pth extension and add .json 
    print("saving model construction hyperparameters in " + json_save)
    with open(json_save, "w") as outfile:
        json.dump(model_params, outfile)