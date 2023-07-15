from hyperparameters.hyperparameters import Hyperparameters
from training import get_dataclass
from ax.service.utils.report_utils import exp_to_df

def get_optimal_hyperparameters_args(dataclass, args):
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
    assert args.unsupervised == False, "Hyperparameters only work for supervised models"

    hype_PATH = "saved_hypes/DVAE" + "_" + args.transition + "_" \
        + args.measurement + "_" + args.inference + "_" \
        + args.encoder + "_" + args.initializer + "_" \
        + args.dataset + ".json"

    hypes_exp = Hyperparameters(dataclass, const_hypes)
    hypes_exp =  hypes_exp.load(hype_PATH)
    hypes_outcomes = exp_to_df(hypes_exp.experiment)
    params_df = hypes_outcomes[hypes_outcomes["trial_index"]==args.trial]

    # reset the arguments to match the optimised hyperparameters 
    args.lr = params_df["lr"].item()
    args.bs = params_df["bs"].item()
    args.T = params_df["T"].item()
    args.stride = params_df["stride"].item()
    args.L2 = params_df["L2"].item()
    args.hdim = params_df["hdim"].item()