import os
import torch
import json
import time
from sklearn.metrics import f1_score
from kan import *
from my_utils import *

# TODO:
# CODE TO LEARN LOSS TOO
# Here: learnable sb bases? and biases? other bx that is better for this?
# What is actually the best way to do this model?
#   - combine each dim of x and w to form a 500-dim vector per class and then do a KANm per vector? 
#           (2*500 + 500*1 learned splines --> repeat across classes)
#   - always learn 500*2 as a kernel with dilatation on the second half
#   - learn a final LAN layer to add to the model as it is that can compute a final combination function
#   between x and each class vector (how to implement this?)

## Grid updates don't keep somme % of the previous grid points... they change everything according to 
## the train batch at the moment of the update... Shouldn't this be changed to allow for a more smooth
## training and to train better with batch!=-1, to not completely just forget the previous batches ????????????

''' Metrics for BCEwithLogitsLoss() '''
def train_f1(pred, target):
    return f1_score(
            target.flatten().cpu().numpy(),
            torch.sigmoid(pred).round().flatten().detach().cpu().numpy(),
            average="macro",
            zero_division=0.0,
        )

def test_f1():
    return f1_score(
            dataset["test_label"].flatten().cpu().numpy(),
            torch.sigmoid(model(dataset["test_input"])).round().flatten().detach().cpu().numpy(), #model(dataset["test_input"])
            average="macro",
            zero_division=0.0,
        )

''' Metrics for the other loss function for with_GAP datasets'''
def train_error(pred, target):
    return torch.mean(
        torch.abs(pred - target).float()
        )

def test_error():
    return torch.mean(
        torch.abs(model(dataset["test_input"]) - dataset["test_label"]).float()
        )

''' Metrics for the other loss function for non-gap datasets'''
def train_acc(pred, target):
    return torch.mean(
        (
            torch.round(pred).argmax(-1)
            == target.argmax(-1)
        ).float()
    )

def test_acc():
    return torch.mean(
        (
            torch.round(model(dataset["test_input"])).argmax(-1)
            == dataset["test_label"].argmax(-1)
        ).float()
    )



losses = {
    "RMSE": None,
    "MSE": torch.nn.MSELoss(reduction="mean"),
    "BCE": torch.nn.BCEWithLogitsLoss(reduction="mean"),
}

data_mode = "vecreps"
data_dir = f"/home/carolina/Anansi/MA/KG/MNIST/data/{data_mode}_dataset/"
loss_fn_name = (
    "BCE"  # MSE loss has to be the same as the loss used to train the mlp (model_name)
)
# from which the logits dataset is made
criterion = losses[loss_fn_name]  # torch.nn.BCEWithLogitsLoss(reduction="mean")
with_GAP = False  # train a dataset with normal imgs and gaps

trials = {
    "model_names": ["Gonzalo_MLP4_SIGMOID_BCE"],
    "n_classes": 10 - 2,
    "base_fun": [
        torch.nn.SiLU(),
        torch.nn.Sigmoid(),
        #torch.nn.Tanh(),
        #Sin(),
    ],
    "grid": [20],  # [10, 20],
    "k": 3,
    #"grid_eps": 1, #controls how much the grid gets updated with a uniform grid between input_range 
                    #and (1-eps)*with actual G input_values from the batch
    "loss_fn": criterion,
    "batch": 128,
    "opt": ["Adam"],  # "LBFGS", Adam
    "lr": 0.01,
    "grid_update_num": 25, #how many times you want to update the grid from samples
    "stop_grid_update_step": 100, #at which step to stop updating the grid
            # therefore, it updates the grid at 0 step and at every int(stop_grid_update_step/grid_update_num) steps 
    "lamb": 0.5,
    "lamb_l1": 1,
    "lamb_entropy": 2,
    "steps": 10, #100, #epochs
}

#CHANGE MY UTILS DATASET FUNC
trial_ini_number = 0

if __name__ == "__main__":

    for model_name in trials["model_names"]:

        model_dir = (
            os.path.join(os.getcwd(), f"results/myKANmodels/{loss_fn_name}/{model_name}")
            if not with_GAP
            else os.path.join(os.getcwd(), f"results/myKANmodels/{loss_fn_name}/{model_name}_wGAP")
        )
        os.makedirs(model_dir, exist_ok=True)

        dataset = create_mydataset(data_dir, model_name, data_mode, w_gap=with_GAP)
        d_model = dataset["train_input"][0].shape[-1]

        trial_num = trial_ini_number
        for bx in trials["base_fun"]:
            for num_grid_point in trials["grid"]:
                model = ClassVecConv(
                    grid=num_grid_point,
                    k=trials["k"],
                    base_fun=bx,
                    n_classes=trials["n_classes"],
                    d_model=d_model,
                    w_path=os.path.join(data_dir, f"{model_name}__classifierweights.pt")
                    # bias_trainable=trials["bias_trainable"],
                    # sp_trainable=trials["sp_trainable"],
                    # sb_trainable=trials["sb_trainable"],
                    # allweights_sharing=trials["allweights_sharing"],
                )

                params, trainable_params = count_parameters(model)
                print(f"Number of parameters: {params}; Number of trainable params: {trainable_params}")
                # grid point values are not trainable params, but are updated from samples during training!!!!

                # train
                for optim in trials["opt"]:

                    save_dir = os.path.join(model_dir, f"{str(trial_num)}")
                    os.makedirs(save_dir, exist_ok=True)

                    print(f"Running model {model_name}, trial={trial_num}")
                    if trials["batch"] != -1:
                        steps = trials["steps"] * int(dataset["train_input"].shape[0]/trials["batch"])

                    if loss_fn_name=="BCE":
                        metrics=(train_f1, test_f1) 
                    elif with_GAP:
                        metrics=(train_error, test_error)
                    else:
                        metrics=(train_acc, test_acc)
                    trials["metrics"] = (metrics[0].__name__, metrics[1].__name__)

                    tini = time.time()
                    results = model.train(
                        dataset,
                        opt=optim,
                        lr=trials["lr"],
                        steps=steps,
                        batch=trials["batch"],
                        loss_fn=trials["loss_fn"],
                        lamb=trials["lamb"],
                        lamb_l1=trials["lamb_l1"],
                        lamb_entropy=trials["lamb_entropy"],
                        update_grid=True,
                        save_fig=False,
                        img_folder=save_dir,
                        grid_update_num=trials["grid_update_num"],
                        stop_grid_update_step=trials["stop_grid_update_step"],
                        metrics=metrics,
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

                    model.plot(
                        beta=3,
                        tick=True,
                        plot_full=False,
                        folder=os.path.join(save_dir, "figures"),
                    )

                    trial_num += 1
