from imports import *
from functions import *
import imports
import functions
import importlib
importlib.reload(imports)
importlib.reload(functions)
import numpy as np

def get_activation(name):
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

class PINN(nn.Module):
    """
    PINN for u_t = u_xx on x in [0,1], t in [0,T],
    with u(x,0) = sin(pi x), u(0,t) = u(1,t) = 0.

    Trial solution:
        g(x,t) = (1 - t) sin(pi x) + x (1 - x) t N_theta(x,t)
    """

    def __init__(self, layers, activation="tanh", device=None):
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

    # Raw network N_theta(x,t)
    def network(self, x, t):
        inp = torch.cat([x, t], dim=1)  # (N,2)
        return self.net(inp)            # (N,1)

    # Initial condition u(x,0) = sin(pi x)
    def u0(self, x):
        return torch.sin(torch.pi * x)

    # Trial solution g(x,t)
    def trial_solution(self, x, t):
        N = self.network(x, t)
        return (1.0 - t) * self.u0(x) + x * (1.0 - x) * t * N

    # Right-hand side f(x,t) = 0 here
    def f(self, x, t):
        return torch.zeros_like(x)

    # PDE residual r = g_t - g_xx - f
    def residual(self, x, t):
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
        r = self.residual(x_coll, t_coll)
        return torch.mean(r**2)

    def train_pinn(self, x_coll, t_coll,
                   epochs=5000, lr=1e-3,
                   optimizer_cls=optim.Adam,
                   verbose_every=500):
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

