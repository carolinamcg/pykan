import torch
import torch.nn as nn
import numpy as np
from .KAN import *
from .LBFGS import *
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO:
# I might have to do a LAN layer at the end for each combination of c_logit and x_logit,
#otherwise, the c_logits will always be fixed for every x and I don't know if this will be
#enought to learn properly


class ClassVecInteractor(nn.Module):
    def __init__(
        self,
        grid,
        k,
        base_fun,
        n_classes,
        d_model,
        w_path,
        device="cpu",
        seed=0,
    ) -> None:
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.input_encoder = KAN(
            width=[d_model, 1],
            grid=grid,
            k=k,
            noise_scale=0.1,
            noise_scale_base=0.1,
            base_fun=base_fun,
            symbolic_enabled=True,
            bias_trainable=False,
            grid_eps=1.0,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            device=device,
            seed=seed,
            allweights_sharing=False,
            LAN=False,
        )

        self.classifier_encoder = KAN(
            width=[d_model, 1],
            grid=grid,
            k=k,
            noise_scale=0.1,
            noise_scale_base=0.1,
            base_fun=base_fun,
            symbolic_enabled=True,
            bias_trainable=False,
            grid_eps=1.0,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            device=device,
            seed=seed,
            allweights_sharing=False,
            LAN=False,
        )

        if w_path is not None:
            # initialize it with fixed weights learned by a pre-trained model
            self.classifier = nn.Parameter(torch.load(w_path), requires_grad=False).to(
                device
            )
        else:
            # learn the classifier weights as well
            self.classifier = nn.Parameter(
                torch.zeros(n_classes, d_model, dtype=torch.float64), requires_grad=True
            ).to(device)

    def initialize_classifier(self, w_path):
        self.classifier = torch.load(w_path)

    def forward(self, x):
        """
        trying to learn the best combination between x and each class vector here
        """
        x_logit = self.input_encoder(x)  # (batch, 1)
        c_logits = self.classifier_encoder(self.classifier)  # (n_classes,1)
        final_logits = x_logit.unsqueeze(1) + c_logits  # (batch, n_classes, 1)
        # the final predictions for each class are the combination of data representation x and the class_vector representation
        del x_logit, c_logits, x
        return final_logits.squeeze(-1)

    def update_grid_from_samples(self, x):
        """
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (batch, input dimension)

        Returns:
        --------
            None
        """
        self.input_encoder.update_grid_from_samples(x)
        self.classifier_encoder.update_grid_from_samples(self.classifier)

    def train(
        self,
        dataset,
        opt="LBFGS",
        steps=100,
        log=1,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        update_grid=True,
        grid_update_num=10,
        loss_fn=None,
        lr=1.0,
        stop_grid_update_step=50,
        batch=-1,
        small_mag_threshold=1e-16,
        small_reg_factor=1.0,
        metrics=None,
        sglr_avoid=False,
        save_fig=False,
        in_vars=None,
        out_vars=None,
        beta=3,
        save_fig_freq=1,
        img_folder="./video",
        device="cpu",
    ):
        """
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            batch : int
                batch size, if -1 then full.
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large factos
            device : str
                device
            save_fig_freq : int
                save figure every (save_fig_freq) step

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization

        Example
        -------
        >>> # for interactive examples, please see demos
        >>> from utils import create_dataset
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.1, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.train(dataset, opt='LBFGS', steps=50, lamb=0.01);
        >>> model.plot()
        """

        def reg(encoder):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.0
            for i in range(len(encoder.acts_scale)):
                vec = self.input_encoder.acts_scale[i].reshape(
                    -1,
                )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = -torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(encoder.act_fun)):
                coeff_l1 = torch.sum(
                    torch.mean(torch.abs(encoder.act_fun[i].coef), dim=1)
                )
                coeff_diff_l1 = torch.sum(
                    torch.mean(torch.abs(torch.diff(encoder.act_fun[i].coef)), dim=1)
                )
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        pbar = tqdm(range(steps), desc="description", ncols=100)

        loss_name = None  # CHANGED
        if loss_fn == None:  # loss RMSE
            loss_name = "RMSE"  # CHANGED
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(
                self.parameters(),
                lr=lr,
                history_size=10,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-32,
                tolerance_change=1e-32,
                tolerance_ys=1e-32,
            )

        results = {}
        results["train_loss"] = []
        results["test_loss"] = []
        results["reg"] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset["train_input"].shape[0]:
            batch_size = dataset["train_input"].shape[0]
            batch_size_test = dataset["test_input"].shape[0]
            nb_batches=1
        else:
            batch_size = batch
            batch_size_test = batch
            nb_batches = int(dataset["train_input"].shape[0]/batch_size) #CHANGED

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset["train_input"][train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(
                    pred[id_], dataset["train_label"][train_id][id_].to(device)
                )
            else:
                train_loss = loss_fn(pred, dataset["train_label"][train_id].to(device))
            reg_ = reg(self.input_encoder) + reg(self.classifier_encoder)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for _ in pbar:  # steps

            # selects batch_size random ids from the full dataset, 1x per step. So, if we want to
            # have a batch_size!=-1 (not the full dataset), we need to increase the number of steps
            # (epochs*num_batches)
            train_id = np.random.choice(
                dataset["train_input"].shape[0], batch_size, replace=False
            )
            test_id = np.random.choice(
                dataset["test_input"].shape[0], batch_size_test, replace=False
            )

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(
                    dataset["train_input"][train_id].to(device)
                )

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset["train_input"][train_id].to(device))
                target = dataset["train_label"][train_id]
                if sglr_avoid == True:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(
                        pred[id_], target[id_].to(device)
                    )
                else:
                     train_loss = loss_fn(
                        pred, target.to(device)
                    )
                reg_ = reg(self.input_encoder) + reg(self.classifier_encoder)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(
                self.forward(dataset["test_input"][test_id].to(device)),
                dataset["test_label"][test_id].to(device),
            )

            if _ % log == 0:
                pbar.set_description(
                    "train loss: %.2e | test loss: %.2e | reg: %.2e "
                    % (
                        torch.sqrt(train_loss).cpu().detach().numpy(),
                        torch.sqrt(test_loss).cpu().detach().numpy(),
                        reg_.cpu().detach().numpy(),
                    )
                )

            if metrics != None and _%nb_batches==0: #save it only when the equivalent to one epoch has passed
                for i in range(len(metrics)):
                    if "train" in metrics[i].__name__:
                        results[metrics[i].__name__].append(metrics[i](pred, target).item()) #train acc for the batch in question
                    else:
                        results[metrics[i].__name__].append(metrics[i]().item())

            if loss_name == "RMSE":
                results["train_loss"].append(
                    torch.sqrt(train_loss).cpu().detach().item()
                )
                results["test_loss"].append(torch.sqrt(test_loss).cpu().detach().item())
            else:
                results["train_loss"].append(train_loss.cpu().detach().item())
                results["test_loss"].append(test_loss.cpu().detach().item())
            results["reg"].append(reg_.cpu().detach().item())

            if save_fig and _ % save_fig_freq == 0:
                self.plot(
                    folder=img_folder,
                    in_vars=in_vars,
                    out_vars=out_vars,
                    title="Step {}".format(_),
                    step=str(_),
                    beta=beta,
                    plot_full=False,
                )
                plt.savefig(
                    img_folder + "/" + str(_) + ".jpg", bbox_inches="tight", dpi=200
                )
                plt.close()

        return results

    def plot(
        self,
        folder="./figures",
        step="",
        beta=3,
        mask=False,
        mode="supervised",
        scale=0.5,
        tick=True,
        sample=False,
        in_vars=None,
        out_vars=None,
        title=None,
        plot_full=False,
    ):
        os.makedirs(folder, exist_ok=True)
        self.input_encoder.plot(
            folder=os.path.join(folder, 'x'),
            beta=beta,
            mask=False,
            mode=mode,
            scale=scale,
            tick=tick,
            sample=sample,
            in_vars=in_vars,
            out_vars=out_vars,
            title=title,
            plot_full=plot_full,
        )
        if plot_full:
            plt.savefig(folder + "/" + step + ".jpg", bbox_inches="tight", dpi=200)
            plt.close()

        self.classifier_encoder.plot(
            folder=os.path.join(folder, 'w'),
            beta=beta,
            mask=False,
            mode=mode,
            scale=scale,
            tick=tick,
            sample=sample,
            in_vars=in_vars,
            out_vars=out_vars,
            title=title,
            plot_full=plot_full,
        )
        if plot_full:
            plt.savefig(folder + "/" + step + ".jpg", bbox_inches="tight", dpi=200)
            plt.close()

    def save_ckpt(self, name, folder="./model_ckpt"):
        """
        save the current model as checkpoint

        Args:
        -----
            name: str
                the name of the checkpoint to be saved
            folder : str
                the folder that stores checkpoints

        Returns:
        --------
            None
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.state_dict(), folder + "/" + name)
        print("save this model to", folder + "/" + name)

    def load_ckpt(self, name, folder="./model_ckpt"):
        """
        load a checkpoint to the current model

        Args:
        -----
            name: str
                the name of the checkpoint to be loaded
            folder : str
                the folder that stores checkpoints

        Returns:
        --------
            None
        """
        self.load_state_dict(torch.load(folder + "/" + name))
