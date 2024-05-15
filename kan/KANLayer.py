import torch
import torch.nn as nn
import numpy as np
from .spline import *

'''
Note:
1. Weight Sharing
    - They're approach initializes and updates the grid and coeffs of every spline as if no weight sharing exists,
    and then just utilizes the idxs defined in self.weight_sharing to compute the post-spline activations. 
    If self.weight_sharing is all 0's, the only spline being used is the 0th one. But, since the updates during training
    ignore this, the spline being used is only updated based on input values that correspond to the 1st spline 
    (x[0, :], where x has shape=(spline, batch)).
    - This sounds unreasonable if you want to learn an activation function that generalizes across in_neurons, for example,
    cause the splines are ot being updated based on all the neurons, but just considering one of them (idx=0) and after that
    gets applied to all neurons that share it...
    - So, they're approach seems ok for just applying and testing, but not if you're intention is to actually learn a function
    that can generalize to all neurons (it might still work though, cause the coeffs and scale_base are trainable by backprop)

    - My approach for learning the same activation fucntion (spline) across neurons for LAN=True is activated by allweights_sharing=True.
    It initializes only one (spline) grid and coeff and applies this to all the neurons. Then, when updating these, instead of
    just using the 0th neurons input values, I flatten x and use all its values (neurons*batch) to update my grid and coeff.
    - Despite having only 1 gid and coeff vectors (less parameters), every implementation I tried does not make it faster than
    tha normal training (allweights_sharing=False).
    - But, at least, it seems to learn the same function for all neurons (apart for changes in post-spline ranges, do to the scale_base.
    If you want this to be completely equal, you have to set it to 1) 
    - This is only implemented for LAN (1-to-1 neuron connections). For KAN, it makes sense to learn in_neurons splines for all out_neurons
    (e.g.: weight_sharing= [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]). Thus, I shall do something ~convolution kernel for this one.
'''



class KANLayer(nn.Module):
    """
    KANLayer class


    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device

    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(
        self,
        in_dim=3,
        out_dim=2,
        num=5,
        k=3,
        noise_scale=0.1,
        scale_base=1.0,
        scale_sp=1.0,
        base_fun=torch.nn.SiLU(),
        grid_eps=0.02,
        grid_range=[-1, 1],
        sp_trainable=True,
        sb_trainable=True,
        device="cpu",
        allweights_sharing=False,
        LAN=False,
    ):
        """'
        initialize a KANLayer

        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            LAN: bool #MY CHANGE
                LAN mode, learnable activation on the nodes instead of on the weights

        Returns:
        --------
            self

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        (3, 5)
        """
        super(KANLayer, self).__init__()
        # size
        if not LAN:  # CHANGED
            self.out_dim = out_dim
        else:
            self.out_dim = 1

        self.size = size = self.out_dim * in_dim  # CHANGED
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.LAN = LAN
        #CHANGED
        self.allweights_sharing = allweights_sharing if LAN else False #can't be true if LAN is False
        if self.allweights_sharing: #CHANGED
            n_different_splines = 1
            self.weight_sharing = torch.zeros(size).long()
            self.lock_counter = 1
            self.lock_id = torch.ones(
                size
            )  # set to 1 on the positions of the respective splines that were locked (sharing weights)
            # but this is just used then if we want to unlock them
        else:
            n_different_splines = size
            self.weight_sharing = torch.arange(size)
            self.lock_counter = 0
            self.lock_id = torch.zeros(
                size
            )  # set to 1 on the positions of the respective splines that were locked (sharing weights)
            # but this is just used then if we want to unlock them

        # shape: (size, num) or (1, num) if only 1 spline should be learned and shared (num=G)
        self.grid = torch.einsum(
            "i,j->ij",
            torch.ones(n_different_splines, device=device), #CHANGED
            torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device),
        )
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        noises = (torch.rand(n_different_splines, self.grid.shape[1]) - 1 / 2) * noise_scale / num #CHANGED
        noises = noises.to(device)
        # shape: (size, coef)=(size,G+k)
        self.coef = torch.nn.Parameter(
            curve2coef(self.grid, noises, self.grid, k, device)
        )
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(
                torch.ones(size, device=device) * scale_base
            ).requires_grad_(
                sb_trainable
            )  # make scale trainable
        else:
            self.scale_base = torch.nn.Parameter(
                torch.FloatTensor(scale_base).to(device)
            ).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(
            torch.ones(size, device=device) * scale_sp
        ).requires_grad_(
            sp_trainable
        )  # make scale trainable
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(
            False
        )
        self.grid_eps = grid_eps
        # self.weight_sharing = torch.arange(size)
        # self.lock_counter = 0
        # self.lock_id = torch.zeros(
        #     size
        # )  # set to 1 on the positions of the respective splines that were locked (sharing weights)
        # # but this is just used then if we want to unlock them
        self.device = device

    def forward(self, x):
        """
        KANLayer forward given input x

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        (torch.Size([100, 5]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]))
        """
        batch = x.shape[0]
        # x: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x = (
            torch.einsum("ij,k->ikj", x, torch.ones(self.out_dim, device=self.device))
            .reshape(batch, self.size)
            .permute(1, 0)
        )
        preacts = x.permute(1, 0).clone().reshape(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x).permute(1, 0)  # shape (batch, size)
        y = coef2curve(
            x_eval=x,
            grid=self.grid[self.weight_sharing],
            coef=self.coef[self.weight_sharing],
            k=self.k,
            device=self.device,
        )  # shape (size, batch)
        # spline that are sharing weights have the same index value in self.weight_sharing (shape=(out_dim, in_dim))
        # if this happens, the initialized self.coef and self.grid positions that are not indexed in self.weight_sharing,
        # won't be trained; and the same position/index of these will be trained to fit more than 1 weight --> sharing
        # the same spline function
        y = y.permute(1, 0)  # shape (batch, size)
        postspline = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        if not self.LAN:  # CHANGED
            y = torch.sum(
                y.reshape(batch, self.out_dim, self.in_dim), dim=2
            )  # shape (batch, out_dim)
        # if LAN, don't sum the splines of each in dimensions. Cause these will be the outpu of each node

        # y shape: (batch, out_dim); preacts shape: (batch, in_dim, out_dim)
        # postspline shape: (batch, in_dim, out_dim); postacts: (batch, in_dim, out_dim)
        # postspline is for extension; postacts is for visualization
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x):
        """
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        """
        # #CHANGED
        if self.allweights_sharing: #allweights_sharing across splines... I'm going to update the 1 grid being trained
                        #according to the largest range seen across all splines/input neuron values
            x = x.flatten().unsqueeze(0)
            batch = x.shape[1] #input from the training dataset
        else:
            batch = x.shape[0] #input from the training dataset
            x = (
                torch.einsum(
                    "ij,k->ikj",
                    x,
                    torch.ones(
                        self.out_dim,
                    ).to(self.device),
                )
                .reshape(batch, self.size)
                .permute(1, 0)
            )  # (size, batch)

        x_pos = torch.sort(x, dim=1)[0] #for each row/spline, sorts the inputs in the batch by ascending order of its values per spline
        y_eval = coef2curve(
            x_pos, self.grid, self.coef, self.k, device=self.device
        )  # (size, batch)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch/ num_interval * i) for i in range(num_interval)] + [
            -1
        ]  # uses only num_interval+1 samples to update the grid
        grid_adaptive = x_pos[:, ids]
        # if self.allweights_sharing: #allweights_sharing across splines... I'm going to update the 1 grid being trained
        #                  #according to the largest range seen across all splines/input neuron values
        #     #grid_adaptive = grid_adaptive.mean(axis=0).unsqueeze(0) #OR choose one of the in_neurons randomly?
        #     #grid_adaptive = grid_adaptive[np.random.randint(0,x.shape[0]), :].unsqueeze(0)
        #     s_ids = torch.argsort(grid_adaptive.abs(), dim=0)
        #     grid_adaptive = grid_adaptive[s_ids[-1, :], torch.arange(num_interval+1)].unsqueeze(0)
        margin = 0.01
        # this is going to update the grod according to the range of inputs seen in train data
        grid_uniform = torch.cat(
            [
                grid_adaptive[:, [0]] #the minimum x value for each spline
                - margin
                + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a
                for a in np.linspace(0, 1, num=self.grid.shape[1])
            ],
            dim=1,
        )
        self.grid.data = (
            self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        ) #new grid will be updated to a grid_uniforms by eps and kept according to the input_range (non_uniform) by 1-eps
        self.coef.data = curve2coef(
            x_pos, y_eval, self.grid, self.k, device=self.device
        )

    def initialize_grid_from_parent(self, parent, x):
        """
        update grid from a parent KANLayer & samples

        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        """
        batch = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x_eval = (
            torch.einsum(
                "ij,k->ikj",
                x,
                torch.ones(
                    self.out_dim,
                ).to(self.device),
            )
            .reshape(batch, self.size)
            .permute(1, 0)
        )
        x_pos = parent.grid
        sp2 = KANLayer(
            in_dim=1,
            out_dim=self.size,
            k=1,
            num=x_pos.shape[1] - 1,
            scale_base=0.0,
            device=self.device,
        )
        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1, device=self.device)
        y_eval = coef2curve(
            x_eval, parent.grid, parent.coef, parent.k, device=self.device
        )
        percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
        self.grid.data = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k, self.device)

    def get_subset(self, in_id, out_id):
        """
        get a smaller KANLayer from a larger KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : KANLayer

        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        """
        spb = KANLayer(
            len(in_id),
            len(out_id),
            self.num,
            self.k,
            base_fun=self.base_fun,
            device=self.device,
        )
        spb.grid.data = self.grid.reshape(self.out_dim, self.in_dim, spb.num + 1)[
            out_id
        ][:, in_id].reshape(-1, spb.num + 1)
        spb.coef.data = self.coef.reshape(self.out_dim, self.in_dim, spb.coef.shape[1])[
            out_id
        ][:, in_id].reshape(-1, spb.coef.shape[1])
        spb.scale_base.data = self.scale_base.reshape(self.out_dim, self.in_dim)[
            out_id
        ][:, in_id].reshape(
            -1,
        )
        spb.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[out_id][
            :, in_id
        ].reshape(
            -1,
        )
        spb.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][
            :, in_id
        ].reshape(
            -1,
        )

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        spb.size = spb.in_dim * spb.out_dim
        return spb

    def lock(self, ids):
        """
        lock activation functions to share parameters based on ids

        Args:
        -----
            ids : list
                list of ids of activation functions

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        """
        self.lock_counter += 1
        # ids: [[i1,j1],[i2,j2],[i3,j3],...]
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = (
                    ids[0][1] * self.in_dim + ids[0][0]
                )
                # it makes the postion in self.weight_sharing correspondng to (x,j), with x=ids[i][0] and j =ids[i][j],
                # have the same weight_sharing id has the first (x,j) pair on ids (ids[0])
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = self.lock_counter

    def unlock(self, ids):
        """
        unlock activation functions

        Args:
        -----
            ids : list
                list of ids of activation functions

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.unlock([[0,0],[1,2],[2,1]]) # unlock the locked functions
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        """
        # check ids are locked
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]]
                == self.weight_sharing[ids[0][1] * self.in_dim + ids[0][0]]
            )
        if locked == False:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = (
                ids[i][1] * self.in_dim + ids[i][0]
            )
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1
