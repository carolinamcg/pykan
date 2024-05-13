import os
import torch
import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sympy.printing.repr import srepr
from kan import *

def create_mydataset(data_dir, model_name):
    assert os.path.exists(
        os.path.join(data_dir, f"{model_name}__Xlogitsdata.pt")
    ), "You have to create the logits dataset for this model first!"
    logits = torch.load(os.path.join(data_dir, f"{model_name}__Xlogitsdata.pt"))
    targets = torch.load(os.path.join(data_dir, f"{model_name}__Ylogitsdata.pt"))
    X_train, X_test, y_train, y_test = train_test_split(
        logits, targets, test_size=0.2, random_state=42
    )
    dataset = {}
    dataset["train_input"] = X_train
    dataset["train_label"] = y_train
    dataset["test_input"] = X_test
    dataset["test_label"] = y_test
    return dataset


def get_best_trials(model, loss_fn):
    acc, loss = -1, 100
    max_acc_trial, min_loss_trial = None, None

    model_dir = os.path.join(os.getcwd(), f"results/{loss_fn}/{model}")
    all_trials = os.listdir(model_dir)

    for trial in all_trials:
        f = open(os.path.join(model_dir, f"{trial}/results.json"), "r")
        results = json.loads(f.read())

        best_acc = np.asanyarray(results[f"{metric}_acc"]).max()
        if best_acc > acc:
            acc = best_acc
            max_acc_trial = (trial, best_acc, results[f"{metric}_loss"][-1])

        best_loss = np.asanyarray(results[f"{metric}_loss"]).min()
        if best_loss < loss:
            loss = best_loss
            min_loss_trial = (trial, best_loss, results[f"{metric}_acc"][-1])

    return max_acc_trial, min_loss_trial


def get_best_models(model_names, loss_names):
    best_trials = {}
    for model in model_names:
        best_trials[model] = {}
        for loss_fn in loss_names:
            best_trials[model][loss_fn] = {"Acc": None, "Loss": None}

            max_acc_trial, min_loss_trial = max_acc_trial, min_loss_trial

            best_trials[model][loss_fn][
                "Acc"
            ] = f"trial_{max_acc_trial[0]}, acc={max_acc_trial[1]}, final_loss={max_acc_trial[2]}"
            best_trials[model][loss_fn][
                "Loss"
            ] = f"trial_{min_loss_trial[0]}, loss={min_loss_trial[1]}, final_loss={min_loss_trial[2]}"

    return best_trials


loss_names = ["MSE"]  # "RMSE",
model_names = ["Gonzalo_MLP4_SIGMOID"]
metric = "train"

#symbolic 
loss = "MSE"
model_name = "Gonzalo_MLP4_SIGMOID"
data_dir = "/home/carolina/Anansi/MA/KG/MNIST/data/logits_dataset/"

base_act_fun = {"SiLU()": torch.nn.SiLU(), "Sigmoid()": torch.nn.Sigmoid(), "Tanh()": torch.nn.Tanh()}

if __name__ == "__main__":
    #best_trials = get_best_trials(model_names, loss_names)
    #print(best_trials)

    # load dataset to get the model's pre and post-spline activations in order
    # to compute the symbolic formulas that better approximate each learned activation

    dataset = create_mydataset(data_dir, model_name)

    best_model_trials = get_best_trials(model_name, loss)
    for trial in best_model_trials:
        trial_dir = os.path.join(os.getcwd(), f"results/{loss}/{model_name}/{trial[0]}")

        args_dir = os.path.join(trial_dir, "args.json")  # args
        with open(os.path.join(trial_dir, "args.json"), "r") as fa:
            params = json.load(fa)
            print(params)
        fa.close()

        model = KAN(
            width=[params["n_classes"], params["n_classes"]],
            grid=params["grid"],
            k=params["k"],
            seed=0,
            base_fun=base_act_fun[params["base_fun"]],
            symbolic_enabled=True,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
            LAN=params["LAN"],
        )

        model.load_ckpt('ckpt1', folder=trial_dir)
        model(dataset['train_input'])

        tini = time.time()
        model.auto_symbolic() #lib=lib
        formulas, _ = model.symbolic_formula()
        tout = time.time()
        print(f"Auto-symbolic took {tout-tini}s")

        formulas_str = []
        for eq in formulas:
            formulas_str.append(srepr(eq))

        with open(os.path.join(trial_dir, "symbolic_formulas.json"), "w") as f:
            json.dump(formulas_str, f, indent=4)
        f.close()


"""
{'Gonzalo_MLP4_SIGMOID': {
    'RMSE': {
        'Acc': 'trial_7, acc=0.9973841905593872, final_loss=0.028523304658958526', 
        'Loss': 'trial_10, loss=0.026439979730385425, final_loss=0.9971464276313782'
        }, 
    'MSE': {
        'Acc': 'trial_11, acc=0.9973577857017517, final_loss=0.0008010898449726167', #nn.Tanh, grid=20, opt=Adam
        'Loss': 'trial_10, loss=0.0006993270458868492, final_loss=0.9971728324890137' #nn.Tanh, grid=20, opt=LBFGS
        }
    }
}

args of trial_10 (MSE) with bias, sp and sp_scalable=True was slightly better!

Gonzalo_MLP4_SIGMOID, MSE, trial 10:
0:
0.5*tanh(1.29*x_1 + 0.03) + 0.5
1:
0.5*tanh(0.15*x_2 - 0.31) + 0.5
2:
1.01 - 1.01*sigmoid(0.39 - 0.26*x_3)
3:
1.0*sigmoid(0.35*x_4 - 0.08)
4:
0.51*tanh(0.15*x_5 - 0.4) + 0.51
5:
0.5*tanh(0.16*x_6 - 0.8) + 0.5
6:
0.5*tanh(0.16*x_7 - 0.08) + 0.5
7:
0.32*atan(1.66*x_8 + 0.14) + 0.5
"""
