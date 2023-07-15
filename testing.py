import argparse
import json
import time 
import os 

import torch 
import torch.distributions as dists 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sb 

from data.dataprep import DataPrep
from data.utils import sliding_window, win_to_seq
from torch_dvae.measurement_models import * 
from torch_dvae.transition_models import * 
from torch_dvae.encoder_models import * 
from torch_dvae.inference_models import * 
from torch_dvae.initializers import * 
from torch_dvae.DVAE import DVAE 
from torch_dvae.utils import score_func, alpha_coverage, alpha_mean
from training import transition_selector, measurement_selector, inference_selector, \
                    init_selector, encoder_selector, select_dims, prep_data, str2bool
from semi_supervised import test_semisupervised_model, make_model, SemiSupervisedModel

sb.set_theme()  # super important 
# --------------------------------------------------
#                   Testing Method 
# --------------------------------------------------
def test_model(dvae: DVAE, test_x, test_t, test_y, T, N, device, unsupervised=False):
    """
    Given the trained DVAE model, testing data and time window this method will generate 
    the latent and input variables based on the conditional variables (or no conditional variables 
    in the unsupervised case) along with multiple metrics used to test the model performance and 
    the mean and standard deviations of the variables for plotting/visualizing the model outputs. 

    The output is a dictionary of all the relevant variables needed in the testing plot functions 
    described within this script and the metrics that are reported to the user. 

    Inputs:
        dvae (DVAE class): the trained DVAE model 
        test_x (list[tensor]): a list of tensors containing the conditional variables used in the
            DVAE model. e.g. the sensor variables of a turbofan engine, each entry in the list 
            is from a different engine unit. 
        test_t (list[tensor]): the list of corresponding time/cycles related to the conditional variables
        test_y (list[tensor]): the list of corresponding input variable tensors related to the conditional 
            variables e.g. for the turbofan engine example this could be the remaining useful life 
            at the corresponding engine cycle. 
        T (int): The time window. A sliding time window is applied to the test data and the model is 
            applied to each time window in parallel (as the model is a sequence to sequence model).
            Technically any time window can be used (doesn't have to be the time window trained on);
            however, it is generally the case for these noncausal models that using the training time window 
            results in the best performance (this should be further explored in future work). 
        N (int): Number of samples. The DVAE model can evaluate multiple samples to simulate multiple possible 
            trajectories. These samples are sampling from p(y_{1:T}|x_{1:T}) and hence, this is what allows 
            the DVAE to quantify the uncertainty of the inputs (y) given the conditional variables (x). 
            Note the mean and standard deviation is returned in the results output so that plotting the bounds 
            and mean trajectory can be done. 
            
            *Note: If the outputs do not approximately follow a Gaussian the mean and standard deviation will not 
            be a good representation of the outputs. Hence, the samples are returned so the user may check this 
            and show the model can represent the uncertainty through the samples themselves with a custom 
            non-analytical distribution. 

        device (torch.device): "cpu" or "gpu" torch device. Using "gpu" is faster as the opperations are done in PyTorch 
            and can be done in parallel. 
        unsupervised (boolean): If no conditional variable was used set this to True and let test_x=test_y (conditional 
            variables = input variables) and this will test how well the model can reconstruct these inputs. 

    Outputs:
        results (dict): A dictionary of the output variables and metrics used to test the DVAE model.
            Note some metrics like "alpha-coverage" and "score" are explained in the "utils file in the torch_dvae directory 
            and list the papers that explain them in further detial. 
    """
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
        "times": [] 
    }
    with torch.no_grad():
        MSE = [] 
        NLL = [] 
        scores = [] 
        cov_95 = [] 
        cov_90 = [] 
        cov_50 = []
        mu_95 = [] 
        mu_90 = []
        mu_50 = [] 
        T = int(T)
        for i, x in enumerate(test_x):
            y = test_y[i][0].to(device).float()
            x = x[0].to(device).float()
            t = test_t[i][0,:,0].to(device).float()

            # --- get time windowed data --- 
            x = sliding_window(x, T)
            y = sliding_window(y, T)

            # --- generate inputs and latent variables ---
            if unsupervised:
                z_dist, y_dist = dvae.reconstruct(y, x)
                zs = z_dist.sample([N]) 
                ys = y_dist.sample([N])
            else:
                _, _, zs, ys = dvae.noncausal_forward(x, N)

                z_dist = dists.normal.Normal(zs.mean(0), zs.std(0))
                y_dist = dists.normal.Normal(ys.mean(0), ys.std(0))
                
            coverage_95 = alpha_coverage(ys[:,:,0], y[:,0], 0.95)
            coverage_90 = alpha_coverage(ys[:,:,0], y[:,0], 0.9)
            coverage_50 = alpha_coverage(ys[:,:,0], y[:,0], 0.5)
            mean_95 = alpha_mean(ys[:,:,0], 0.95)
            mean_90 = alpha_mean(ys[:,:,0], 0.9)
            mean_50 = alpha_mean(ys[:,:,0], 0.5)

            z_mean = z_dist.loc     
            z_stds = z_dist.scale   
            y_mean = y_dist.loc     
            y_stds = y_dist.scale   

            # --- convert back to seq ---
            x = win_to_seq(x)
            y = win_to_seq(y)
            z_mean = win_to_seq(z_mean)
            z_stds = win_to_seq(z_stds)
            y_mean = win_to_seq(y_mean)
            y_stds = win_to_seq(y_stds)
            zs = win_to_seq(zs, 1, 2).permute(1,0,2)   # (N, seq, dim)
            ys = win_to_seq(ys, 1, 2).permute(1,0,2)   
            coverage_95 = win_to_seq(coverage_95)
            coverage_90 = win_to_seq(coverage_90)
            coverage_50 = win_to_seq(coverage_50)
            mean_95 = win_to_seq(mean_95)
            mean_90 = win_to_seq(mean_90)
            mean_50 = win_to_seq(mean_50)

            y_dist = dists.normal.Normal(y_mean, y_stds)
            # --- get losses --- 
            nll = -y_dist.log_prob(y).sum(0).mean()
            mse = (y_mean - y) ** 2
            score = score_func(y_mean[-1,:], y[-1])

            # --- store variables --- 
            MSE.append(mse.detach().cpu().numpy())
            NLL.append(nll.detach().cpu().numpy())
            scores.append(score.detach().cpu().numpy())
            cov_95.append(coverage_95.detach().cpu().numpy())
            cov_90.append(coverage_90.detach().cpu().numpy())
            cov_50.append(coverage_50.detach().cpu().numpy())
            mu_95.append(mean_95.detach().cpu().numpy())
            mu_90.append(mean_90.detach().cpu().numpy())
            mu_50.append(mean_50.detach().cpu().numpy())
            results["y_true"].append(y.detach().cpu().numpy())
            results["y_mean"].append(y_mean.detach().cpu().numpy())
            results["y_stds"].append(y_stds.detach().cpu().numpy())
            results["z_mean"].append(z_mean.detach().cpu().numpy())
            results["z_stds"].append(z_stds.detach().cpu().numpy())
            results["zs"].append(zs.detach().cpu().numpy())
            results["ys"].append(ys.detach().cpu().numpy())
            results["times"].append(t.detach().cpu().numpy())

    MSE = np.concatenate(MSE, axis=0)
    RMSE = np.sqrt(MSE.mean())
    results["RMSE"] = RMSE 

    nll = sum(NLL) / len(NLL)   # mean nll over all units 
    results["y_nll"] = nll 

    scores = np.concatenate(scores, axis=0).sum()
    results["score"] = scores 

    cov_95 = np.concatenate(cov_95, axis=0).mean()
    cov_90 = np.concatenate(cov_90, axis=0).mean()
    cov_50 = np.concatenate(cov_50, axis=0).mean()
    mu_95 = np.concatenate(mu_95, axis=0).mean()
    mu_90 = np.concatenate(mu_90, axis=0).mean()
    mu_50 = np.concatenate(mu_50, axis=0).mean()
    results["alpha_cover_95"] = cov_95
    results["alpha_cover_90"] = cov_90
    results["alpha_cover_50"] = cov_50
    results["alpha_mean_95"] = mu_95
    results["alpha_mean_90"] = mu_90
    results["alpha_mean_50"] = mu_50

    return results 
# --------------------------------------------------
#                   Plotting 
# --------------------------------------------------
def plot_sensors(unit, results):
    """
    Used for high dimensional DVAE inputs usually for sensor reconstruction plots 
    when using unsupervised training 
    """ 
    true = results["y_true"][unit-1]                  
    ests = results["y_mean"][unit-1] 
    stds = results["y_stds"][unit-1]  
    upper = ests + 3*stds 
    lower = ests - 3*stds

    seq = true.shape[1]
    t = results["times"][unit-1]
    ests = ests[:,:seq]
    upper = upper[:,:seq]
    lower = lower[:,:seq]

    sensors = true.shape[-1]
    dim = int(np.ceil(np.sqrt(sensors)))
    fig, axes = plt.subplots(dim, dim, figsize=(22,22))
    i = 0
    j = 0

    for sensor in range(sensors):
        if i == dim:
            i = 0 
            j += 1

        axes[i,j].plot(t, true[:,sensor], color="tab:red", label="True")                      
        axes[i,j].plot(t, ests[:,sensor], color="tab:blue", label="Estimates") 
        axes[i,j].plot(t, upper[:,sensor], ls='--', color='k')        
        axes[i,j].plot(t, lower[:,sensor], ls='--', color='k')        
        axes[i,j].fill_between(t, upper[:,sensor], lower[:,sensor], color='k', alpha=0.2) 
        axes[i,j].set_xlabel("cycles")                                  
        axes[i,j].set_ylabel("normalized sensor values")                
        axes[i,j].legend()                                              
        axes[i,j].set_title("sensor {}".format(sensor+1))               

        i += 1 
    plt.show()

def plot_rul_vs_time(unit, results):
    """
    Plots the rul vs time for a single unit (if your DVAE outputs are RUL values)
    """
    t = results["times"][unit-1]
    lower_bound = results["y_mean"][unit-1][:,0] - results["y_stds"][unit-1][:,0]*2
    upper_bound = results["y_mean"][unit-1][:,0] + results["y_stds"][unit-1][:,0]*2

    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.plot(t, results["y_mean"][unit-1], label="mean RUL estimate")
    plt.fill_between(t, upper_bound, lower_bound, alpha=0.3, label="95$\%$ confidence interval")
    plt.plot(t, results["y_true"][unit-1], lw=2, label="true RUL", color="tab:red")

    #plt.title("Unit %i: RUL vs Time"%unit, fontsize=20)
    plt.xlabel("Time (cycles)", fontsize=32)
    plt.ylabel("RUL (cycles)", fontsize=32)
    plt.legend(prop={"size": 28})
    plt.show()

def get_final_ruls(results):
    """
    Gets and stores the final test time RUL estimate (the latest estimate the model made)
    which is often used in evaluating prognostic models 

    *Note the DVAE needs to be trained to output RUL values. Don't use this if unsupervised learning 
    was used for sensor reconstruction 

    Inputs:
        results (dict): results from the testing method 
    
    Outputs: 
        rmse (array): RMSE of the final RUL estimate vs the true RUL 
        r_final (array): final/latest true RUL for each unit in the testing dataset 
        r_fin_est (array): final/latest estimated mean RUL from the model for each unit 
        r_fin_std (array): final/latest estimated standard deviation of the RUL for each unit 
        max_time (int): the maximum time value considering all the testing units (used later in plotting)
    """
    r_final = [] 
    r_fin_est = [] 
    r_fin_std = [] 

    mse = 0 
    units = len(results["y_true"])
    max_time = 0

    for i in range(units):
        r_max = results["y_true"][i][0,0]
        r_fin = results["y_true"][i][-1,0]
        r_est = results["y_mean"][i][-1,0]
        r_std = results["y_stds"][i][-1,0]

        mse += float(np.mean((r_fin - r_est) ** 2))
        r_final.append(r_fin)
        r_fin_est.append(r_est)
        r_fin_std.append(r_std)

        if r_max > max_time:
            max_time = r_max

    mse = mse / units
    rmse = np.sqrt(mse)
    r_final = np.stack(r_final)
    r_fin_est = np.stack(r_fin_est)
    r_fin_std = np.stack(r_fin_std)
    return rmse, r_final, r_fin_est, r_fin_std, max_time

def plot_final_rul_vs_time(r_final, r_fin_est, max_time):
    """
    Plots the final/latest rul estimate for each unit in the testing dataset with 
    time on the x-axis and contrasts it with the true final RUL vs time (which would be 
    a linear equation with gradient of 1 and y-intercept at 0). 

    Hence, how well the final RUL estimates (plotted as a scatter plot) track this line
    representing the True RUL, is a good visual indicator for the accuracy of the model.
    Often we expect the lower the time value the better the estimate (the closer we are to failure 
    the better the RUL estimate, as the data better represents the machines imminent failure)
    """
    t = np.linspace(0, max_time+1, 1000)
    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.plot(t, t, color="k", lw=2., label="true")
    plt.scatter(r_fin_est, r_final, label="estimates")

    plt.xlabel("RUL Estimates (cycles)", fontsize=32)
    plt.ylabel("True RUL (cycles)", fontsize=32)
    plt.legend(prop={"size": 28})
    plt.show()

def plot_final_rul_vs_units(r_final, r_fin_est, r_fin_std):
    """
    Plots the final/latest RUL estimate with respect to the unit/machine (as the x-axis). 
    It also shows the bounds calculated with the standard deviation so we can see if the RUL estimate 
    is within these bounds and how tight they are. 

    This plot shows us an overall picture of how well the bounds capture the uncertainties in the model
    and also show us which units were easy with regards to RUL estimation and which were difficult. 
    """
    units = r_final.shape[0]
    unit_list = np.linspace(1, units, units)
    bound = 2*r_fin_std

    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.scatter(unit_list, r_final, marker="o", color="tab:red", label="true")   
    plt.errorbar(unit_list, r_fin_est, bound, capsize=5., fmt="o", label="estimates")

    plt.xlabel("Unit Number", fontsize=32)
    plt.ylabel("RUL (cycles)", fontsize=32)
    plt.legend(prop={"size": 28})
    plt.show()

def plot_latent_vs_time(unit, results):
    """
    Plots the latent trajectories vs time for a specific unit 
    """
    z_mean = results["z_mean"][unit-1]
    z_stds = results["z_stds"][unit-1]
    t = results["times"][unit-1]

    plt.figure(figsize=(18,9))
    plt.plot(t, z_mean, label="latent")

    for i in range(z_mean.shape[-1]):
        plt.fill_between(t, 
                        z_mean[...,i] + 2*z_stds[...,i],
                        z_mean[...,i] - 2*z_stds[...,i],
                        alpha=0.4)

    plt.xlabel("Time (cycles)", fontsize=32)
    plt.ylabel("$\mathbf{z}$", fontsize=32) 
    plt.show()

def plot_latent_phase_space(unit, results, dim1, dim2):
    """
    Plots the 2D phase space of the choosen latent dimensions for a specific unit 
    (only works for model with 2 or more latent dimensions)
    """
    z_mean = results["z_mean"][unit-1]
    t = results["times"][unit-1]

    plt.figure(figsize=(18,9))
    plt.scatter(z_mean[:,dim1], z_mean[:,dim2], c=t)
    plt.xlabel("$z_1$", fontsize=32)
    plt.ylabel("$z_2$", fontsize=32)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.colorbar()
    plt.show()

def plot_latent_phase_space_all(results, dim1, dim2):
    ts = np.concatenate(results["times"])
    zs = np.concatenate(results["z_mean"])

    seq = 0 
    z_mean = 0
    for i, z in enumerate(results["z_mean"]):
        l = z.shape[0]
        y = results["y_true"][i][-1,0]
        if seq < l:
            seq = l
            z_mean = z 
            t_mean = results["times"][i]
            
    plt.figure(figsize=(18,9))
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.plot(z_mean[:,dim1], z_mean[:,dim2], color="tab:blue", lw=3)     
    plt.scatter(zs[:,dim1], zs[:,dim2], c=ts)

    plt.colorbar()
    plt.xlabel("$z_1$", fontsize=32)
    plt.ylabel("$z_2$", fontsize=32)
    plt.show()


def define_model(model_params, 
                 transition, measurement, inference, initializer, encoder,
                 controls=True, semi_supervised=False):
        
        xdim = model_params["xdim"]
        hdim = model_params["hdim"]
        zdim = model_params["zdim"]
        ydim = model_params["ydim"]
        transition_inputs = model_params["transition inputs"]
        measurement_inputs = model_params["measurement inputs"]
        init_inputs = model_params["initializer inputs"]

        if semi_supervised:
            unsupervised_transition_inputs = model_params["unsupervised transition inputs"]
            unsupervised_measurement_inputs = model_params["unsupervised measurement inputs"]
            unsupervised_initializer_inputs = model_params["unsupervised initializer inputs"]
            unsupervised_transition = model_params["unsupervised transition"]
            unsupervised_measurement = model_params["unsupervised measurement"]
            unsupervised_inference = model_params["unsupervised inference"]
            unsupervised_initializer = model_params["unsupervised initializer"]
            unsupervised_encoder = model_params["unsupervised encoder"]

            unsupervised_model = make_model(xdim, hdim, zdim, xdim, unsupervised_transition, unsupervised_measurement, unsupervised_inference, unsupervised_initializer,
                        unsupervised_encoder, unsupervised_transition_inputs, unsupervised_measurement_inputs, unsupervised_initializer_inputs, has_controls=False)

            model = make_model(xdim+zdim, hdim, zdim, ydim, transition, measurement, inference, initializer, encoder, 
                           transition_inputs, measurement_inputs, init_inputs, controls)
            model = SemiSupervisedModel(unsupervised_model, model)

        else:
            model = make_model(xdim, hdim, zdim, ydim, transition, measurement, inference, initializer, encoder, 
                           transition_inputs, measurement_inputs, init_inputs, controls)

        return model 
# -----------------------------------------------------------
#                       Run Tests 
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--save_path", type=str, default="saved_models/DVAE")
    parser.add_argument("--save_results", type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument("--run_model", type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument("--plot_unit", type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument("--transition", type=str, default="mlp")
    parser.add_argument("--measurement", type=str, default="mlp")
    parser.add_argument("--inference", type=str, default="rnn")
    parser.add_argument("--encoder", type=str, default="rnn")
    parser.add_argument("--initializer", type=str, default="controls")
    parser.add_argument("--best", type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument("--controls", type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument("--unsupervised", type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--split", type=float, default=0)
    parser.add_argument("--semisupervised", type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument("--N", type=int, default=100)
    args = parser.parse_args()

    # --- Get Testing Data ---
    PATH = "CMAPSS"
    prep_class = DataPrep(PATH, args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "FD001" or args.dataset == "FD003":
        prep_class.op_normalize(K=1)    # K=1 normalization, K=6 operating condition norm 
    else: 
        prep_class.op_normalize(K=6)

    x_test, y_test, t_test = prep_class.prep_test(prep_class.ntest, prep_class.RUL)

    if args.unsupervised:
        y_test = x_test 
        path_append = "pre"
    else:
        path_append = ""

    if not args.controls:
        args.initializer = "measure"    # can't use control variables if they don't exist; use measurements instead 

    # change file name if we are testing the current best model 
    if args.best == True:
        save_path = args.save_path + path_append + "_best"
    else:
        save_path = args.save_path + path_append

    save_PATH = save_path + "_" + args.transition + "_" \
                + args.measurement + "_" + args.inference + "_" \
                + args.encoder + "_" + args.initializer + "_" \
                + args.dataset + ".pth"

    if args.semisupervised:
        save_PATH = save_PATH[:-4] + "_semi" + "_" + str(args.split) + ".pth"

    # --- Run Model --- 
    if args.run_model == True:
        # --- Define Model --- 
        with open(save_PATH[:-4] + ".json") as file:
            model_params = json.load(file)

        print("Loading hyperparameter .json file: " + save_PATH[:-4] + ".json")
        T = model_params["T"]

        model = define_model(model_params, args.transition, args.measurement, args.inference, args.initializer, args.encoder,
                            args.controls, args.semisupervised)

        model.load_state_dict(torch.load(save_PATH))
        model = model.to(device)

        # --- Testing ---
        begin = time.time()
        if args.semisupervised:
            results = test_semisupervised_model(model, x_test, t_test, y_test, T, args.N, device)
        else:
            results = test_model(model, x_test, t_test, y_test, T, args.N, device, unsupervised=args.unsupervised)  
        end = time.time()
        runtime = end - begin

        print(f"Total testing runtime: {runtime/60} minutes")
        # --- Save Results ---
        if args.save_results:
            npy_save = save_PATH[:-4] + "_test_results" + ".npy"   
            print("saving test results in " + npy_save)
            np.save(npy_save, results, allow_pickle=True)

    # --- Load Results (if run_model is False) --- 
    else:
        npy_save = save_PATH[:-4] + "_test_results" + ".npy"
        assert os.path.exists(npy_save), f"{npy_save}, File does not exist"
        results = np.load(npy_save, allow_pickle=True).tolist() # actually returns a dict. when calling tolist

    # --- Print and Plot Results --- 
    print("Total RMSE: ", results["RMSE"])
    print("Total -log-likelihood: ", results["y_nll"])
    print("Total score: ", results["score"])
    print("95% coverage ", results["alpha_cover_95"])
    print("90% coverage ", results["alpha_cover_90"])
    print("50% coverage ", results["alpha_cover_50"])
    print("95% mean width ", results["alpha_mean_95"])
    print("90% mean width ", results["alpha_mean_90"])
    print("50% mean width ", results["alpha_mean_50"])

    if args.plot_unit:
        unit = args.unit
        if args.unsupervised:
            plot_sensors(unit, results)
        else:
            plot_rul_vs_time(unit, results)
            rmse, r_final, r_fin_est, r_fin_std, max_time = get_final_ruls(results)
            plot_final_rul_vs_time(r_final, r_fin_est, max_time)
            plot_final_rul_vs_units(r_final, r_fin_est, r_fin_std)

        plot_latent_vs_time(unit, results)
        plot_latent_phase_space(unit, results, 0, 1)
        plot_latent_phase_space_all(results, 0, 1)