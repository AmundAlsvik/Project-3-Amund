import numpy as np
from classes import PINN

def forward_Euler(dx, T, alpha=0.5, L=1.0):
    """
    Forward Euler for u_t = u_xx on [0,L] with u(x,0)=sin(pi x), u(0,t)=u(L,t)=0.
    Uses dt = alpha * dx^2, so dt/dx^2 = alpha (<= 0.5 for stability).

    Parameters
    ----------
    dx    : float
        Spatial step size.
    T     : float
        Final time to integrate to.
    alpha : float, default 0.5
        Ratio dt/dx^2 (stability requires alpha <= 0.5).
    L     : float, default 1.0
        Length of the rod.

    Returns
    -------
    x : (Nx,) array
    t : (Nt,) array
    U : (Nt, Nx) array, U[n,i] ≈ u(x_i, t_n)
    """
    Nx = int(L / dx) + 1
    x = np.linspace(0.0, L, Nx)

    dt = alpha * dx**2
    Nt = int(T / dt) + 1
    t = np.linspace(0.0, T, Nt)

    U = np.zeros((Nt, Nx))
    U[0, :] = np.sin(np.pi * x)
    U[0, 0] = 0.0
    U[0, -1] = 0.0

    u_xx = np.zeros_like(U[0, :])
    u_t  = np.zeros_like(U[0, :])

    for n in range(Nt - 1):
        u = U[n, :]

        u_xx = np.zeros_like(u)
        u_t  = np.zeros_like(u)

        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

        u_t[1:-1] = u_xx[1:-1]

        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + dt * u_t[1:-1]

        u_new[0]  = 0.0
        u_new[-1] = 0.0

        U[n+1, :] = u_new



    return x, t, U


def analytical_diffusion(x,t):
    """
    Returns
    -------
    array-like
        analytical solution of 1D Heat diffusion
    """
    return np.exp((-np.pi**2)*t)*np.sin(np.pi*x)


def u(x):
    return np.sin(np.pi*x)

def g_trial(point,P):
    x,t = point
    return (1-t)*u(x) + x*(1-x)*t*PINN(P,point)


def f(point):
    return 0


def jacobian_cost(P, x, t):
    cost_sum = 0
    from autograd import jacobian,hessian,grad

    g_t_jacobian_func = jacobian(g_trial)
    g_t_hessian_func = hessian(g_trial)

    for x_ in x:
        for t_ in t:
            point = np.array([x_,t_])

            g_t = g_trial(point,P)
            g_t_jacobian = g_t_jacobian_func(point,P)
            g_t_hessian = g_t_hessian_func(point,P)

            g_t_dt = g_t_jacobian[1]
            g_t_d2x = g_t_hessian[0][0]

            func = f(point)

            err_sqr = ( (g_t_dt - g_t_d2x) - func)**2
            cost_sum += err_sqr

    return cost_sum /( np.size(x)*np.size(t) )
