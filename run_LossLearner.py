import os
import torch
import json
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from my_utils import *
from kan import *

def train_f1(pred, target):
    return f1_score(
            target.flatten().cpu().numpy(),
            torch.round(pred).flatten().detach().cpu().numpy(),
            average="macro",
            zero_division=0.0,
        )

def test_f1():
    return f1_score(
            dataset["test_label"].flatten().cpu().numpy(),
            torch.round(model(dataset["test_input"])).flatten().detach().cpu().numpy(),
            average="macro",
            zero_division=0.0,
        )


losses = {
    "RMSE": None,
    "MSE": torch.nn.MSELoss(reduction="mean"),
    "BCE": torch.nn.BCEWithLogitsLoss(reduction="mean"),
}

data_dir = "/home/carolina/Anansi/MA/KG/MNIST/data/logits_dataset/"
loss_fn_name = (
    "MSE"  # MSE loss has to be the same as the loss used to train the mlp (model_name)
)
criterion = losses[loss_fn_name]
with_GAP = False  # train a dataset with normal imgs and gaps

trials = {
    "model_names": ["Gonzalo_MLP4_SIGMOID"],
    "n_classes": 10 - 2,
    "base_fun": [
        #torch.nn.SiLU(),
        torch.nn.Sigmoid(),
        #torch.nn.Tanh(),
        #Sin(),
    ],
    "grid": [20],  # [10, 20],
    "k": 3,
    "bias_trainable": False,  # if this is false, the bias = 0
    "sp_trainable": False,  # if false, this is 1 (which multiplies by the spline(x))
    "sb_trainable": False,  # if false, this is set to scale_base (see KAN init) (which multiplies by the bx(x))
    "allweights_sharing": True,
    "loss_fn": criterion,
    "opt": ["Adam"],  # "LBFGS", Adam
    "grid_update_num": 20, #how many times you want to update the grid from samples
    "stop_grid_update_step": 100, #at which step to stop updating the grid
            # therefore, it updates the grid at 0 step and at every int(stop_grid_update_step/grid_update_num) steps 
    "lamb": 0.1,
    "lamb_l1": 1,
    "lamb_entropy": 2,
}

trial_ini_number = 0

if __name__ == "__main__":

    for model_name in trials["model_names"]:

        model_dir = (
            os.path.join(os.getcwd(), f"results/LossLearner/{loss_fn_name}/{model_name}")
            if not with_GAP
            else os.path.join(os.getcwd(), f"results/LossLearner/{loss_fn_name}/{model_name}_wGAP")
        )
        os.makedirs(model_dir, exist_ok=True)

        dataset = create_mydataset(data_dir, model_name, data_mode="logits", w_gap=with_GAP)

        trial_num = trial_ini_number
        for bx in trials["base_fun"]:
            for num_grid_point in trials["grid"]:
                model = KAN(
                    width=[trials["n_classes"], trials["n_classes"], trials["n_classes"]], #1st layer = activation function, 2nd layer = loss function
                    grid=num_grid_point,
                    k=trials["k"],
                    seed=0,
                    base_fun=bx,
                    symbolic_enabled=True,
                    bias_trainable=trials["bias_trainable"],
                    sp_trainable=trials["sp_trainable"],
                    sb_trainable=trials["sb_trainable"],
                    allweights_sharing=trials["allweights_sharing"],
                    LAN=True,
                )

                # if trials["weight_sharing"] and LAN:
                #     ids_list = [[i, j] for j in range(1) for i in range(trials["n_classes"])]
                #     model.lock(0, ids_list) #layer 0

                params, trainable_params = count_parameters(model)
                print(f"Number of parameters: {params}; Number of trainable params: {trainable_params}")
                # grid point values are not trainable params, but are updated from samples during training!!!!

                # train
                for optim in trials["opt"]:

                    save_dir = os.path.join(model_dir, f"{str(trial_num)}")
                    os.makedirs(save_dir, exist_ok=True)

                    steps = 400  # 50 if optim == "LBFGS" else 100
                    print(f"Running model {model_name}, trial={trial_num}")
                    tini = time.time()
                    results = model.train(
                        dataset,
                        opt=optim,
                        steps=steps,
                        batch=-1,
                        loss_fn=trials["loss_fn"],
                        update_grid=True,
                        save_fig=False,
                        img_folder=save_dir,
                        grid_update_num=trials["grid_update_num"],
                        stop_grid_update_step=trials["stop_grid_update_step"],
                        metrics=(train_acc, test_acc),
                    )

                    tout = time.time()
                    print(f"Training time={tout-tini}s")

                    # save ckpt
                    model.save_ckpt("ckpt1", folder=save_dir)

                    # save args
                    args = copy.deepcopy(trials)
                    args["base_fun"] = str(bx)
                    args["grid"] = num_grid_point
                    args["opt"] = optim
                    args["steps"] = steps
                    args["loss_fn"] = loss_fn_name
                    with open(os.path.join(save_dir, "args.json"), "w") as f:
                        json.dump(args, f, indent=4)  # 'indent=4' for pretty printing

                    # save results
                    # results_to_save = {key: value.tolist() for key, value in results.items()}
                    with open(os.path.join(save_dir, "results.json"), "w") as f:
                        json.dump(results, f, indent=4)

                    # save loss plot
                    save_loss_plot(results, save_dir)


                    model.plot_LAN(
                        beta=7,
                        tick=True,
                        plot_full=True,
                        folder=os.path.join(save_dir, "figures"),
                    )

                    plt.savefig(os.path.join(save_dir, "figures") + "fullmodel.jpg", bbox_inches="tight", dpi=200)
                    plt.close()

                    trial_num += 1
