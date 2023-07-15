import argparse
import json
import time 

import torch 
import torch.nn as nn
import numpy as np 

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient, ObjectiveProperties
import ax

from data.dataprep import DataPrep
from torch_dvae.measurement_models import * 
from torch_dvae.transition_models import * 
from torch_dvae.encoder_models import * 
from torch_dvae.inference_models import * 
from torch_dvae.initializers import * 
from torch_dvae.DVAE import DVAE 
from semi_supervised import make_model
from training import Trainer, str2bool

class Hyperparameters(nn.Module):
    def __init__(self, DataFramework, const_hypes):
        """
        Inputs: 
            DataFramework (class): DataPrep class (normalize before inputting)
        """
        super().__init__()
        self.const_hypes = const_hypes
        self.data = DataFramework

    def bayes_opt(self, trials, filename, load_previous=False): 
        """
        Uses the Ax framework to perform Bayesian Optimization. This is an optimization
        technique which can find the optimum hyperparameters in a more efficient manner 
        than grid and random search through the use of surrogate functions. 

        Inputs:
            trials (int): number of trials/hyperparameters the bayesian optimizer will test 

            filename (str): string stating the path and filename of the saved hyperparameters 

            load_previous (bool):
                if load_previous == True this specifies that the AxClient should continue 
                optimization until it has completed a total number of trials 
                specified by the input (trials). 
                *AxClient is loaded from the file specified by the "filename" input
        """

        if load_previous == False: 
            self.ax_client = AxClient()

            self.ax_client.create_experiment(
                name="hyperparameter_optimization",
                parameters = [
                    {"name": "lr", "type": "range", "bounds": [1e-4, 5e-3], "log_scale": True, "value_type": "float"}, 
                    {"name": "bs", "type": "range", "bounds": [150, 550], "value_type": "int"},
                    {"name": "T", "type": "range", "bounds": [20, 50], "value_type": "int"},
                    {"name": "stride", "type": "range", "bounds": [1,5], "value_type": "int"}, 
                    {"name": "L2", "type": "range", "bounds": [1e-6, 1e-4], "log_scale": True, "value_type": "float"},
                    {"name": "hdim", "type": "range", "bounds": [50, 500], "value_type": "int"},
                ],
                objectives={"nll": ObjectiveProperties(minimize=True), "mse": ObjectiveProperties(minimize=True)}
            )

        if load_previous: 
            self.ax_client = self.load(filename, verbose_logging=True)
            
        for _ in range(trials):
            parameters, trial_index = self.ax_client.get_next_trial()

            if trial_index >= (trials): # if loading a previous client stop at max trials
                break

            self.ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate(parameters))
            self.save(filename)

        best_parameters = self.ax_client.get_pareto_optimal_parameters()

        return best_parameters

    def evaluate(self, parameters): 
        """
        INPUTS:
            parameters (dictionary): 
                a dictionary that contains [str, float] of the different 
                hyperparameters
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Get hyperparameters ---
        lr = parameters.get("lr", 1e-3)
        L2 = parameters.get("L2", 1e-5)
        T = parameters.get("T", 40)
        bs = parameters.get("bs", 250)
        hdim = parameters.get("hdim", 50)
        zdim = self.const_hypes["zdim"]
        ydim = 1 # RUL dim 
        stride = parameters.get("stride", 1)
        epochs = self.const_hypes["epochs"]
        split = self.const_hypes["split"]
        transition_inputs = self.const_hypes["transition_inputs"]
        measurement_inputs = self.const_hypes["measurement_inputs"]
        init_inputs = self.const_hypes["init_inputs"]
        controls = self.const_hypes["controls"]
        inference = self.const_hypes["inference"]
        transition = self.const_hypes["transition"]
        measurement = self.const_hypes["measurement"]
        initializer = self.const_hypes["initializer"]
        encoder = self.const_hypes["encoder"]

        save_PATH = self.const_hypes["save_path"] + "_" + transition + "_" \
                    + measurement + "_" + inference + "_" \
                    + encoder + "_" + initializer + "_" \
                    + self.data.dataset + ".pth"
        
        # --- prepare data --- 
        x_train, y_train, t_train = self.data.prep_data(self.data.ntrain, T, stride)
        x_train, y_train, t_train, x_valid, y_valid, t_valid = self.data.valid_set(x_train, y_train, t_train, split)
        x_test, y_test, t_test = self.data.prep_test(self.data.ntest, self.data.RUL)
        train_loader, valid_loader = self.data.get_dataloaders(bs, x_train, y_train, x_valid, y_valid)

        xdim = x_train.shape[-1]
        # --- Construct DVAE --- 
        model = make_model(xdim, hdim, zdim, ydim, 
                           transition, measurement, inference, initializer, encoder,
                           transition_inputs, measurement_inputs, init_inputs, controls)
        model = model.to(device)
            
        # --- Model prep and evaluation ---
        torch.save(model.state_dict(), save_PATH)
        trainer = Trainer(lr, L2)

        begin = time.time()
        model = trainer.train_model(epochs, train_loader, valid_loader, model, save_PATH, device, print_loss=False)
        end = time.time()

        runtime = end - begin 

        model_params = {
            "xdim": xdim, 
            "hdim": hdim, 
            "zdim": zdim,
            "ydim": ydim,
            "T": T, 
            "transition inputs": transition_inputs,
            "measurement inputs": measurement_inputs,
            "initializer inputs": init_inputs,
            "bs": bs,
            "lr": lr,
            "L2": L2,
            "valid split": split,
            "train time": runtime,
            "epochs": epochs,
        }

        json_save = save_PATH[:-4] + ".json"    # get rid of .pth extension and add .json 
        print("saving model construction hyperparameters in " + json_save)
        with open(json_save, "w") as outfile:
            json.dump(model_params, outfile)

        valid_nll, valid_nll_std, valid_mse, valid_mse_std = trainer.validate(valid_loader, model, device)

        return {"nll": (valid_nll, valid_nll_std), "mse": (valid_mse, valid_mse_std)}

    def save(self, filepath):
        self.ax_client.save_to_json_file(filepath)

    def load(self, filepath, verbose_logging=False):
        return AxClient(verbose_logging=verbose_logging).load_from_json_file(filepath)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--save_path", type=str, default="saved_models/DVAE")
    parser.add_argument("--hype_path", type=str, default="saved_hypes/DVAE")
    parser.add_argument("--split", type=float, default=0.3)
    parser.add_argument("--zdim", type=int, default=2)
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
    parser.add_argument("--load_previous", type=str2bool, default=False, const=True, nargs="?")
    args = parser.parse_args()

    # store constant hyperparameters we are not optimizing 
    const_hypes = {
        "save_path": args.save_path,
        "epochs": args.epochs,
        "split": args.split,
        "zdim": args.zdim,
        "transition": args.transition, 
        "measurement": args.measurement,
        "inference": args.inference,
        "encoder": args.encoder,
        "initializer": args.initializer,
        "transition_inputs": args.transition_inputs,
        "measurement_inputs": args.measurement_inputs,
        "init_inputs": args.init_inputs,
        "controls": args.controls
    }
    hype_PATH = args.hype_path + "_" + args.transition + "_" \
            + args.measurement + "_" + args.inference + "_" \
            + args.encoder + "_" + args.initializer + "_" \
            + args.dataset + ".json"

    # Load and Prep data 
    PATH = "..\\CMAPSS"
    data = DataPrep(PATH, args.dataset)
    if args.dataset == "FD001" or args.dataset == "FD003":
        data.op_normalize(K=1)    # K=1 normalization, K=6 operating condition norm 
    else: 
        data.op_normalize(K=6) 

    # Hyperparameter optimization 
    hypes = Hyperparameters(data, const_hypes)
    params = hypes.bayes_opt(trials=20, filename=hype_PATH, load_previous=args.load_previous)
