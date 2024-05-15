import os
import torch
import json
import numpy as np
import time
from sklearn.model_selection import train_test_split
from kan import *

##TODO:
## do symbolic for KAN too to see how each class correlates with each other. And check plots to see which ones are "regularized"

## Train with symbolic = False?
## Should I learn the classifier layer as well w/ KAN?

# NOTES:
# 1. I'm using the conda env that I used to train my MLPs and get the (logits) dataset.
# 2. I removed the classe nodes 1 and 9, since my MLPs were not trained with them, they won't ever be predicted
# and so won't appear on this dataset either.
# 3. They say they use "RSME" loss when loss_fn is set to None, but they're actually training on the "MSE" loss value.
# Only after the backward() pass they do the sqrt. (not sure why RMSE got much higher loss than MSE in my results. Might
#                                                   it be the initialization? Or some better bakprop propertie of the pytorch MSE function?)


def create_mydataset(data_dir, model_name, w_gap=False):
    if w_gap:
        Xdata, Ydata = [], []
        for filename in ["train_10000_", "cifar10_10000_"]:
            assert os.path.exists(
                os.path.join(data_dir, f"{model_name}__{filename}Xlogitsdata.pt")
            ), "You have to create the logits dataset for this model first!"
            x = torch.load(os.path.join(data_dir, f"{model_name}__{filename}Xlogitsdata.pt"))
            y = torch.load(os.path.join(data_dir, f"{model_name}__{filename}Ylogitsdata.pt"))
            if "cifar" in filename:
                Xdata.append(x[:1000]) #for the dataset to be a litlle less unbalenced between 0 and 1
                Ydata.append(y[:1000])
            else:
                Xdata.append(x)
                Ydata.append(y)
        logits, targets = stack_datasets(Xdata, Ydata)
    else:
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


def stack_datasets(datasets_Xlist, datasets_Ylist):
    return torch.vstack(datasets_Xlist), torch.vstack(
        datasets_Ylist
    )


def count_parameters(model):
    total_params, total_trainable_params = 0, 0
    for name, param in model.named_parameters():
        print(name, param.shape)
        total_params += param.flatten().shape[0]
        if param.requires_grad:
            total_trainable_params += param.flatten().shape[0]
    return total_params, total_trainable_params

def train_acc():
    return torch.mean(
        (
            torch.round(model(dataset["train_input"])).argmax(-1)
            == dataset["train_label"].argmax(-1)
        ).float()
    )


def test_acc():
    return torch.mean(
        (
            torch.round(model(dataset["test_input"])).argmax(-1)
            == dataset["test_label"].argmax(-1)
        ).float()
    )


def save_loss_plot(results, save_dir):
    """
    Function to save the loss and accuracy plots to disk.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(results["train_loss"], label="train_loss", linestyle="-")
    plt.plot(results["test_loss"], label="test_loss", linestyle="-")

    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


losses = {
    "RMSE": None,
    "MSE": torch.nn.MSELoss(reduction="mean"),
    "BCE": torch.nn.BCEWithLogitsLoss(reduction="mean"),
}
#

data_dir = "/home/carolina/Anansi/MA/KG/MNIST/data/logits_dataset/"
loss_fn_name = (
    "MSE"  # MSE loss has to be the same as the loss used to train the mlp (model_name)
)
# from which the logits dataset is made
criterion = losses[loss_fn_name]  # torch.nn.BCEWithLogitsLoss(reduction="mean")
LAN = True  # if LAN is true, there are only n_classes edges (1 to 1 connections). Therefore,
# not sparse regularixation should be done

with_GAP = True  # train a dataset with normal imgs and gaps

trials = {
    "model_names": ["Gonzalo_MLP4_SIGMOID"],
    "n_classes": 10 - 2,
    "base_fun": [
        #torch.nn.SiLU(),
        torch.nn.Sigmoid(),
        #torch.nn.Tanh(),
        #Sin(),
    ],  # [torch.nn.SiLU(), torch.nn.Sigmoid(), torch.nn.Tanh()],
    # trainable scales and bias?
    "LAN": LAN,
    "grid": [20],  # [10, 20],
    "k": 3,
    "bias_trainable": False,  # if this is false, the bias = 0
    "sp_trainable": False,  # if false, this is 1 (which multiplies by the spline(x))
    "sb_trainable": False,  # if false, this is set to scale_base (see KAN init) (which multiplies by the bx(x))
    "allweights_sharing": True,
    "loss_fn": criterion,
    "opt": ["Adam"],  # "LBFGS", Adam
    "grid_update_num": 100, #how many times you want to update the grid from samples
    "stop_grid_update_step": 100, #at which step to stop updating the grid
            # therefore, it updates the grid at 0 step and at every int(stop_grid_update_step/grid_update_num) steps 
    "lamb": 0.1,
    "lamb_l1": 1,
    "lamb_entropy": 2,
}

trial_ini_number = 33

if __name__ == "__main__":

    prefix = "" if LAN else "KAN_"
    for model_name in trials["model_names"]:

        model_dir = (
            os.path.join(os.getcwd(), f"results/{loss_fn_name}/{model_name}")
            if not with_GAP
            else os.path.join(os.getcwd(), f"results/{loss_fn_name}/{model_name}_wGAP")
        )
        os.makedirs(model_dir, exist_ok=True)

        dataset = create_mydataset(data_dir, model_name, w_gap=with_GAP)

        trial_num = trial_ini_number
        for bx in trials["base_fun"]:
            for num_grid_point in trials["grid"]:
                model = KAN(
                    width=[trials["n_classes"], trials["n_classes"]],
                    grid=num_grid_point,
                    k=trials["k"],
                    seed=0,
                    base_fun=bx,
                    symbolic_enabled=True,
                    bias_trainable=trials["bias_trainable"],
                    sp_trainable=trials["sp_trainable"],
                    sb_trainable=trials["sb_trainable"],
                    allweights_sharing=trials["allweights_sharing"],
                    LAN=LAN,
                )

                # if trials["weight_sharing"] and LAN:
                #     ids_list = [[i, j] for j in range(1) for i in range(trials["n_classes"])]
                #     model.lock(0, ids_list) #layer 0

                params, trainable_params = count_parameters(model)
                print(f"Number of parameters: {params}; Number of trainable params: {trainable_params}")
                # grid point values are not trainable params, but are updated from samples during training!!!!

                # train
                for optim in trials["opt"]:

                    save_dir = os.path.join(model_dir, f"{prefix}{str(trial_num)}")
                    os.makedirs(save_dir, exist_ok=True)

                    steps = 400  # 50 if optim == "LBFGS" else 100
                    print(f"Running model {model_name}, LAN={LAN}, trial={trial_num}")
                    tini = time.time()
                    if LAN:  # no sparsity reguçarization
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
                    else:
                        results = model.train(
                            dataset,
                            opt=optim,
                            steps=steps,
                            batch=-1,
                            loss_fn=trials["loss_fn"],
                            lamb=trials["lamb"],
                            lamb_l1=trials["lamb_l1"],
                            lamb_entropy=trials["lamb_entropy"],
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

                    # save activation functions
                    if LAN:
                        model.plot_LAN(
                            beta=7,
                            tick=True,
                            plot_full=False,
                            folder=os.path.join(save_dir, "figures"),
                        )
                    else:
                        model.plot(
                            beta=7,
                            tick=True,
                            plot_full=False,
                            folder=os.path.join(save_dir, "figures"),
                        )

                    trial_num += 1
