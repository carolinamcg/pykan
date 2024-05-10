import os
import torch
import json
import numpy as np
from sklearn.model_selection import train_test_split
from kan import *

##TODO:
## try with KGAP env
## remove 1 and 9th node or train mls with all 10 numbers
## If I want to use BCELoss here, i have to use sigmoid before... cause it's inputs have to be between 0 and 1

def create_mydataset(data_dir, model_name):
    assert os.path.exists(os.path.join(data_dir, f"{model_name}__Xlogitsdata.pt")), "You have to create the logits dataset for this model first!"
    logits = torch.load(os.path.join(data_dir, f"{model_name}__Xlogitsdata.pt"))
    targets = torch.load(os.path.join(data_dir, f"{model_name}__Ylogitsdata.pt"))
    X_train, X_test, y_train, y_test = train_test_split(logits, targets, test_size=0.2, random_state=42)
    dataset = {}
    dataset['train_input'] = X_train
    dataset['train_label'] = y_train
    dataset['test_input'] = X_test
    dataset['test_label'] = y_test
    return dataset

def count_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        print(name, param.shape)
        total_params += param.flatten().shape[0]
    return total_params

def train_acc():
    return torch.mean((torch.round(model(dataset['train_input'])).argmax(-1) == dataset['train_label'].argmax(-1)).float())

def test_acc():
    return torch.mean((torch.round(model(dataset['test_input'])).argmax(-1) == dataset['test_label'].argmax(-1)).float())

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


losses = {"RMSE": None, "MSE": torch.nn.MSELoss(reduction='mean'), "BCE": torch.nn.BCELoss(reduction='mean')}
#

data_dir = "/home/carolina/Anansi/MA/KG/MNIST/data/logits_dataset/"
loss_fn_name = "RMSE" #MSE loss has to be the same as the loss used to train the mlp (model_name)
                #from which the logits dataset is made
criterion  = losses[loss_fn_name] #torch.nn.BCEWithLogitsLoss(reduction="mean")
LAN = True # if LAN is true, there are only n_classes edges (1 to 1 connections). Therefore,
            # not sparse regularixation should be done
trials = {
    "model_names": ["Gonzalo_MLP4_SIGMOID"],
    "n_classes": 10-2,
    "base_fun": [torch.nn.SiLU(), torch.nn.Sigmoid(), torch.nn.Tanh()],
    # trainable scales and bias?
    "LAN": True,
    "grid": [10, 20],
    "k": 3, #[3, 4],
    "loss_fn": criterion,
    "opt": ["LBFGS", "Adam"],
    "grid_update_num": 10,
    "lamb": 0.1,
    "lamb_l1": 1,
    "lamb_entropy": 2,
}




if __name__ == "__main__":

    for model_name in trials["model_names"]:

        model_dir = os.path.join(os.getcwd(), f"results/{loss_fn_name}/{model_name}")
        os.makedirs(model_dir, exist_ok=True)

        dataset = create_mydataset(data_dir, model_name)

        trial_num = 0
        for bx in trials["base_fun"]:
            for num_grid_point in trials["grid"]:
                model = KAN(width=[trials["n_classes"], trials["n_classes"]], grid=num_grid_point, 
                            k=trials["k"], seed=0, base_fun=bx,
                            symbolic_enabled=True, bias_trainable=False, sp_trainable=False, 
                            sb_trainable=False, LAN=LAN)
                
                print(f"Number of parameters: {count_parameters(model)}")

                #train
                for optim in trials["opt"]:

                    save_dir = os.path.join(model_dir, str(trial_num))
                    os.makedirs(save_dir, exist_ok=True)

                    steps = 50 if optim=="LBFGS" else 100
                    if LAN:
                        results = model.train(dataset, opt=optim, steps=steps, batch=-1, loss_fn=trials["loss_fn"], 
                                            update_grid=True, save_fig=False, img_folder=save_dir,
                                            grid_update_num=10, metrics=(train_acc, test_acc));
                    else:
                        results = model.train(dataset, opt="LBFGS", steps=steps, batch=-1, loss_fn=trials["loss_fn"], 
                                            lamb=trials["lamb"], lamb_l1=trials["lamb_l1"], 
                                            lamb_entropy=trials["lamb_entropy"], update_grid=True,
                                            save_fig=False, img_folder=save_dir,
                                            grid_update_num=10, metrics=(train_acc, test_acc));

                    # save ckpt   
                    model.save_ckpt('ckpt1', folder=save_dir)             

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
                    #results_to_save = {key: value.tolist() for key, value in results.items()}
                    with open(os.path.join(save_dir, "results.json"), "w") as f:
                        json.dump(results, f, indent=4)

                    # save loss plot
                    save_loss_plot(results, save_dir)

                    # save activation functions
                    if LAN:
                        model.plot_LAN(beta=7, tick=True, plot_full=False, folder=os.path.join(save_dir, "figures"))
                    else:
                        model.plot(beta=7, tick=True, plot_full=False, folder=os.path.join(save_dir, "figures"))

                    trial_num += 1
