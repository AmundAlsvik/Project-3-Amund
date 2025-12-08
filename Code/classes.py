from imports import *

def get_activation(name):
    """
    Return a PyTorch activation module based on a string identifier.

    Parameters
    ----------
    name : str or torch.nn.Module
        Name of the activation function ("tanh", "relu", "sigmoid", "gelu"),
        or an already instantiated PyTorch activation module.

    Returns
    -------
    torch.nn.Module
        Corresponding activation function module.

    Raises
    ------
    ValueError
        If the activation name is not recognized.
    """
    if isinstance(name, nn.Module):
        return name
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

def get_optimizer(name, p):
    if isinstance(name, nn.Module):
        return name
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(p.parameters(), lr=lr)
    if name == "rms":
        return torch.optim.RMSprop(p.parameters(), lr=lr, alpha=0.94)
    if name == "ada":
        return torch.optim.Adagrad(p.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(p.parameters(), lr=lr)
    if name == "sgdl2":
        return torch.optim.SGD(p.parameters(), lr=lr, weight_decay=1e-3)
    raise ValueError(f"Unknown optimizer: {name}")


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for the 1D diffusion equation

        u_t = u_xx,

    on the spatial domain x ∈ [0, 1] and time domain t ∈ [0, T], subject to

        u(x, 0) = sin(πx),
        u(0, t) = u(1, t) = 0.

    The solution is represented using a trial function that enforces the
    initial and boundary conditions by construction.

    Trial solution:
        g(x, t) = (1 - t) sin(πx) + x (1 - x) t N_θ(x, t)

    where N_θ(x, t) is a feed-forward neural network.
    """

    def __init__(self, layers, activation="tanh", device=None):
        """
        Initialize the PINN model.

        Parameters
        ----------
        layers : list of int
            Network architecture specified as a list of layer sizes.
            Example: [2, 50, 50, 1].
        activation : str or torch.nn.Module, optional
            Activation function used in hidden layers.
            Default is "tanh".
        device : str or torch.device, optional
            Device to place the model on ("cpu" or "cuda").
            If None, CUDA is used if available.
        """
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        act = get_activation(activation)

        modules = []
        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(act)
        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*modules)
        self.to(self.device)

    def network(self, x, t):
        """
        Evaluate the neural network N_θ(x, t).

        Parameters
        ----------
        x : torch.Tensor
            Spatial coordinates, shape (N, 1).
        t : torch.Tensor
            Temporal coordinates, shape (N, 1).

        Returns
        -------
        torch.Tensor
            Network output N_θ(x, t), shape (N, 1).
        """
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

    def u0(self, x):
        """
        Initial condition u(x, 0) = sin(πx).

        Parameters
        ----------
        x : torch.Tensor
            Spatial coordinates.

        Returns
        -------
        torch.Tensor
            Initial condition evaluated at x.
        """
        return torch.sin(torch.pi * x)

    def trial_solution(self, x, t):
        """
        Construct the trial solution g(x, t).

        The trial solution satisfies all boundary and initial conditions
        exactly.

        Parameters
        ----------
        x : torch.Tensor
            Spatial coordinates.
        t : torch.Tensor
            Temporal coordinates.

        Returns
        -------
        torch.Tensor
            Trial solution g(x, t).
        """
        N = self.network(x, t)
        return (1.0 - t) * self.u0(x) + x * (1.0 - x) * t * N

    def f(self, x, t):
        """
        Right-hand side of the PDE.

        For the diffusion equation considered here, f(x, t) = 0.

        Parameters
        ----------
        x : torch.Tensor
            Spatial coordinates.
        t : torch.Tensor
            Temporal coordinates.

        Returns
        -------
        torch.Tensor
            Right-hand side evaluated at (x, t).
        """
        return torch.zeros_like(x)

    def residual(self, x, t):
        """
        Compute the PDE residual r(x, t) = u_t - u_xx - f(x, t).

        Automatic differentiation is used to evaluate the derivatives.

        Parameters
        ----------
        x : torch.Tensor
            Spatial collocation points.
        t : torch.Tensor
            Temporal collocation points.

        Returns
        -------
        torch.Tensor
            Residual values at collocation points.
        """
        x = x.clone().detach().to(self.device).requires_grad_(True)
        t = t.clone().detach().to(self.device).requires_grad_(True)

        g = self.trial_solution(x, t)

        g_t = torch.autograd.grad(
            g, t,
            grad_outputs=torch.ones_like(g),
            create_graph=True,
            retain_graph=True,
        )[0]

        g_x = torch.autograd.grad(
            g, x,
            grad_outputs=torch.ones_like(g),
            create_graph=True,
            retain_graph=True,
        )[0]

        g_xx = torch.autograd.grad(
            g_x, x,
            grad_outputs=torch.ones_like(g_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        r = g_t - g_xx - self.f(x, t)
        return r

    def loss(self, x_coll, t_coll):
        """
        Compute the physics-informed loss function.

        The loss is defined as the mean squared PDE residual
        over the collocation points.

        Parameters
        ----------
        x_coll : torch.Tensor
            Spatial collocation points.
        t_coll : torch.Tensor
            Temporal collocation points.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        r = self.residual(x_coll, t_coll)
        return torch.mean(r**2)

    def train_pinn(self, x_coll, t_coll,
                   epochs=5000, lr=1e-3,
                   optimizer_cls=optim.Adam,
                   verbose_every=500):
        """
        Train the PINN using gradient-based optimization.

        Parameters
        ----------
        x_coll : torch.Tensor
            Spatial collocation points.
        t_coll : torch.Tensor
            Temporal collocation points.
        epochs : int, optional
            Number of training epochs. Default is 5000.
        lr : float, optional
            Learning rate. Default is 1e-3.
        optimizer_cls : torch.optim.Optimizer, optional
            Optimizer class to use. Default is Adam.
        verbose_every : int, optional
            Print loss every given number of epochs.

        Returns
        -------
        PINN
            The weights and biases of the trained PINN model.
        """
        x_coll = x_coll.reshape(-1, 1).to(self.device)
        t_coll = t_coll.reshape(-1, 1).to(self.device)

        optimizer = optimizer_cls(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss(x_coll, t_coll)
            loss.backward()
            optimizer.step()

            if verbose_every and epoch % verbose_every == 0:
                print(f"Epoch {epoch:5d} | Loss = {loss.item():.3e}")

        return self

if __name__ =="__main__":
        import time
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import seaborn as sns


        device = "cpu" 
        torch.set_default_dtype(torch.float32)

        def u_exact(x, t):
            x = np.asarray(x)
            t = np.asarray(t)
            return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

  
        T_max   = 0.5
        N_coll  = 200       
        epochs  = 1500    
        lr      = 1e-3

        optimizers = ["rms", "ada", "sgd", "sgdl2"]
        activations = ["gelu", "relu", "tanh", "sigmoid"]
        hidden_layers  = [2, 3, 4, 5]      

        mseMatrix  = np.zeros((len(hidden_layers), len(optimizers)))
        timeMatrix = np.zeros_like(mseMatrix)

        Nx_eval, Nt_eval = 40, 40
        x_eval = np.linspace(0.0, 1.0, Nx_eval)
        t_eval = np.linspace(0.0, T_max, Nt_eval)
        Xg, Tg = np.meshgrid(x_eval, t_eval, indexing="ij")
        x_flat = Xg.ravel()
        t_flat = Tg.ravel()
        u_ref  = u_exact(x_flat, t_flat)

 
     
        def train_one_pinn(layers, activation, epochs, lr, opt="adam"):
            pinn = PINN(layers=layers, activation=activation, device=device)
            pinn.to(device)

            param_dtype = next(pinn.parameters()).dtype

     
            x_coll = torch.rand(N_coll, 1, device=device, dtype=param_dtype)
            t_coll = torch.rand(N_coll, 1, device=device, dtype=param_dtype) * T_max

            optimizer = get_optimizer(opt, pinn)

            start = time.perf_counter()
            for ep in range(epochs):
                optimizer.zero_grad()
                loss = pinn.loss(x_coll, t_coll)
                loss.backward()
                optimizer.step()
            end = time.perf_counter()

            train_time = end - start
            return pinn, train_time

        
        for i, h in enumerate(hidden_layers):
            for o, optimizer in enumerate(optimizers):

                print(f"\n=== {h} hidden layers, {128} nodes each ===")

                layers = [2] + [128]*h + [1]  

                pinn, train_time = train_one_pinn(layers, activation="gelu",
                                                epochs=epochs, lr=lr, opt=optimizer)
                timeMatrix[i, o] = train_time


                param_dtype = next(pinn.parameters()).dtype
                with torch.no_grad():
                    x_torch = torch.tensor(x_flat, dtype=param_dtype,
                                        device=device).view(-1, 1)
                    t_torch = torch.tensor(t_flat, dtype=param_dtype,
                                        device=device).view(-1, 1)
                    u_pred  = pinn.trial_solution(x_torch, t_torch).cpu().numpy().ravel()

                mse_val = np.mean((u_pred - u_ref)**2)
                mseMatrix[i, o] = mse_val

                print(f"Time: {train_time:.2f} s | MSE: {mse_val:.2e}")


        fig, axes = plt.subplots(figsize=(11, 4))

        sns.heatmap(
            mseMatrix, annot=True, fmt=".2e", cmap="viridis",
            xticklabels=optimizers, yticklabels=hidden_layers, ax=axes, square=True
        )
        axes.set_xlabel("Optimizer", fontsize=12)
        axes.set_ylabel("Number of hidden layers", fontsize=12)
        axes.set_title("PINN MSE  w/gelu and 128 nodes", fontsize=15)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(figsize=(11, 4))
        sns.heatmap(
            timeMatrix, annot=True, fmt=".2f", cmap="magma",
            xticklabels=optimizers, yticklabels=hidden_layers, ax=axes, square=True
        )
        axes.set_xlabel("Optimizer", fontsize=12)
        axes.set_ylabel("Number of hidden layers", fontsize=12)
        axes.set_title("Training time w/gelu and 128 nodes", fontsize=15)

        plt.tight_layout()
        plt.show()
