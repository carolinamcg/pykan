import os
import torch
from sklearn.model_selection import train_test_split
from kan import *



def create_mydataset(data_dir, model_name, data_mode, w_gap=False):
    if w_gap:
        Xdata, Ydata = [], []
        for filename in ["train_10000_", "cifar10_10000_"]:
            assert os.path.exists(
                os.path.join(data_dir, f"{model_name}__{filename}X{data_mode}data.pt")
            ), "You have to create the logits dataset for this model first!"
            x = torch.load(os.path.join(data_dir, f"{model_name}__{filename}X{data_mode}data.pt"))
            y = torch.load(os.path.join(data_dir, f"{model_name}__{filename}Y{data_mode}data.pt"))
            if "cifar" in filename:
                Xdata.append(x[:1000]) #for the dataset to be a litlle less unbalenced between 0 and 1
                Ydata.append(y[:1000])
            else:
                Xdata.append(x)
                Ydata.append(y)
        logits, targets = stack_datasets(Xdata, Ydata)
    else:
        assert os.path.exists(
            os.path.join(data_dir, f"{model_name}__X{data_mode}data.pt")
        ), "You have to create the logits dataset for this model first!"
        logits = torch.load(os.path.join(data_dir, f"{model_name}__X{data_mode}data.pt"))
        targets = torch.load(os.path.join(data_dir, f"{model_name}__Y{data_mode}data.pt"))

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
