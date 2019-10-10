import numpy as np
import time

eps64 = np.finfo(np.float64).eps

def init_solution(V, domain):
    nx = V.shape[1]
    hx = (domain[1] - domain[0]) / nx
    xc = 0.5 * (domain[0] + domain[1]) + nx / 8 * hx

    V[0, :] = np.abs(domain[0] + (np.arange(nx) + 0.5) * hx - xc) < 0.2
    V[1, :] = 0

    
def model(ff, V):
    ff[0, :] = V[1, :]
    # mask = V[0, :] > eps64
    # ff[1, np.logical_not(mask)] = 0.
    # ff[1, mask] = V[1, mask]**2 / V[0, mask] + 0.5 * 9.81 * V[0, mask]**2
    #ff[1, :] = np.divide(V[1, :]**2, V[0, :], out=np.zeros(ff.shape[1]), where=V[0, :] > eps64) + 0.5 * 9.81 * V[0, :]**2
    ff[1, :] = 0.
    np.divide(V[1, :]**2, V[0, :], out=ff[1, :], where=V[0, :] > eps64)
    ff[1, :] += 0.5 * 9.81 * V[0, :]**2


def LaxFriedrich(V, ff, fluxes, lambdas, max_lambdas, dt, dx):
    nx = V.shape[1]
    Cx = dt / dx

    model(ff, V)
    np.maximum(lambdas[:-1], lambdas[1:], out=max_lambdas)

    Vold_l = V[1, 0]
    Vold_r = V[1, -1]

    np.copyto(fluxes, 0.5 * Cx * ((ff[:, 1:] + ff[:, :-1]) - max_lambdas * (V[:, 1:] - V[:, :-1])))
    V[:, :-1] -= fluxes
    V[:, 1:] += fluxes

    V[1, 0] += Cx * (ff[1, 0] - lambdas[0] * Vold_l)
    V[1, -1] -= Cx * (ff[1, -1] + lambdas[-1] * Vold_r)


def update_eigenvalues(lambdas, V):
    # mask = V[0, :] > eps64
    # lambdas[np.logical_not(mask)] = 0.
    # lambdas[mask] = np.abs(V[1, mask] / V[0, mask]) + np.sqrt(9.81 * V[0, mask])
    #lambdas[:] = np.where(V[0, :] <= eps64, 0, np.abs(V[1, :] / V[0, :]) + np.sqrt(9.81 * V[0, :]))
    #lambdas[:] = np.abs(np.divide(V[1, :], V[0, :], out=np.zeros_like(lambdas), where=V[0, :] > eps64)) + np.sqrt(9.81 * V[0, :])
    lambdas[:] = 0.
    np.divide(np.abs(V[1, :]), V[0, :], out=lambdas, where=V[0, :] > eps64)
    lambdas += np.sqrt(9.81 * V[0, :])
    


def update_to_time(tfinal):
    global t
    while t < tfinal:
        update_eigenvalues(lambdas, V)
        Cmax = np.amax(lambdas)
        dt = min(0.5 * dx / Cmax, tfinal - t)
        LaxFriedrich(V, fmodel, fluxes, lambdas, max_lambdas, dt, dx)
        t += dt

        
        

    

domain = np.array([0, 1])
Nx = 2**12

T = 2.0
dx = (domain[1] - domain[0]) / Nx

t = 0.
V = np.empty((2, Nx))
init_solution(V, domain)

fmodel = np.empty_like(V)
fluxes = np.empty((2, Nx-1))
lambdas = np.empty(Nx)
max_lambdas = np.empty(Nx-1)

    
print(f"Initial time t = {t}")
t_start = time.time()
update_to_time(T)
t_end = time.time()
print(f"End of simulation, t = {t}")
print(f"Elapsed time: {t_end - t_start}")

print(f"mean(h) = {np.mean(V[0, :])}")
np.savetxt("sol-cpu", V[0, :])
    
