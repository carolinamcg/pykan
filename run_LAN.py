import os
import torch
import json
import numpy as np
from sklearn.model_selection import train_test_split
from kan import *

##TODO:
## BCE, than KAN w/ best, than gap+test, than compute symbolic expressions for best
## do symbolic for KAN too to see how each class correlates with each other. And check plots to see which ones are "regularized"

## Train with symbolic = False?
## If I want to use BCELoss here, i have to use sigmoid before... cause it's inputs have to be between 0 and 1
## Train w/ gaps as well know, wil labels all 0
## Should I learn the classifier layer as well w/ KAN?
## Learn the best combination between x and self.classifier? I can only think of this as making 10 vectors
# of 2x the model_dim [x, class_vector] for class_vector in self.classifier for x in batch
# but this would take 4eveeeeer

# NOTES:
# 1. I'm using the conda env that I used to train my MLPs and get the (logits) dataset.
# 2. I removed the classe nodes 1 and 9, since my MLPs were not trained with them, they won't ever be predicted 
# and so won't appear on this dataset either.
# 3. They say they use "RSME" loss when loss_fn is set to None, but they're actually training on the "MSE" loss value.
# Only after the backward() pass they do the sqrt. (not sure why RMSE got much higher loss than MSE in my results. Might
#                                                   it be the initialization? Or some better bakprop propertie of the pytorch MSE function?) 

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


def count_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        print(name, param.shape)
        total_params += param.flatten().shape[0]
    return total_params


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
    "BCE"  # MSE loss has to be the same as the loss used to train the mlp (model_name)
)
# from which the logits dataset is made
criterion = losses[loss_fn_name]  # torch.nn.BCEWithLogitsLoss(reduction="mean")
LAN = True  # if LAN is true, there are only n_classes edges (1 to 1 connections). Therefore,
# not sparse regularixation should be done
trials = {
    "model_names": ["Gonzalo_MLP4_SIGMOID_BCE"],
    "n_classes": 10 - 2,
    "base_fun": [torch.nn.SiLU(), torch.nn.Tanh()], #[torch.nn.SiLU(), torch.nn.Sigmoid(), torch.nn.Tanh()],
    # trainable scales and bias?
    "LAN": LAN,
    "grid": [10], #[10, 20],
    "k": 8,  
    "bias_trainable": False, # if this is false, the bias = 0
    "sp_trainable": False, # if false, this is 1 (which multiplies by the spline(x))
    "sb_trainable": False, # if false, this is 1 (which multiplies by the bx(x))
    "loss_fn": criterion,
    "opt": ["Adam"], #"LBFGS", Adam
    "grid_update_num": 10,
    "lamb": 0.1, 
    "lamb_l1": 1,
    "lamb_entropy": 2,
}

trial_ini_number = 6

if __name__ == "__main__":

    prefix = "" if LAN else "KAN_"
    for model_name in trials["model_names"]:

        model_dir = os.path.join(os.getcwd(), f"results/{loss_fn_name}/{model_name}")
        os.makedirs(model_dir, exist_ok=True)

        dataset = create_mydataset(data_dir, model_name)

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
                    LAN=LAN,
                )

                print(f"Number of parameters: {count_parameters(model)}")

                # train
                for optim in trials["opt"]:

                    save_dir = os.path.join(model_dir, f"{prefix}{str(trial_num)}")
                    os.makedirs(save_dir, exist_ok=True)

                    steps = 100 #50 if optim == "LBFGS" else 100
                    print(f"Running model {model_name}, LAN={LAN}, trial={trial_num}")
                    if LAN: #no sparsity reguçarization
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
                            metrics=(train_acc, test_acc),
                        )

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
