import numpy as np
import numba
import time
import socket


def init_solution(V, domain):
    nx = V.shape[0]
    hx = (domain[1] - domain[0]) / nx
    xc = 0.5 * (domain[0] + domain[1]) + nx / 8 * hx

    V[:, 0] = np.abs(domain[0] + (np.arange(nx) + 0.5) * hx - xc) < 0.2
    V[:, 1] = 0


@numba.njit
def model(ff, V1, V2, tol):
    ff[0] = V2
    if V1 < tol:
        ff[1] = 0.
    else:
        ff[1] = V2**2 / V1 + 0.5 * 9.81 * V1**2


@numba.njit
def LaxFriedrich(V, Vold, lambdas, dt, dx, tol):
    nx = V.shape[0]
    Cx = dt / dx

    flux1 = np.empty(2)
    flux2 = np.empty(2)

    model(flux1, Vold[0, 0], Vold[0, 1], tol)

    V[0, 1] += Cx * (flux1[1] - lambdas[0] * Vold[0, 1])
    for i in range(nx-1):
        model(flux2, Vold[i+1, 0], Vold[i+1, 1], tol)
        lmbd = max(lambdas[i], lambdas[i+1])

        flux = 0.5 * Cx * ((flux2[0] + flux1[0]) - lmbd * (Vold[i+1, 0] - Vold[i, 0]))
        V[i, 0] -= flux
        V[i+1, 0] += flux

        flux = 0.5 * Cx * ((flux2[1] + flux1[1]) - lmbd * (Vold[i+1, 1] - Vold[i, 1]))
        V[i, 1] -= flux
        V[i+1, 1] += flux

        flux1, flux2 = flux2, flux1

    V[-1, 1] -= Cx * (flux1[1] + lambdas[-1] * Vold[-1, 1])
        

@numba.njit
def update_eigenvalues(lambdas, V, tol):
    nx = lambdas.size
    for i in range(nx):
        if V[i, 0] < tol:
            lambdas[i] = 0.
        else:
            lambdas[i] = np.abs(V[i, 1] / V[i, 0]) + np.sqrt(9.81 * V[i, 0])


@numba.njit
def update_to_time(t, tfinal, V, Vold, lambdas, dx, tol):
    while t < tfinal:
        for i in range(V.shape[0]):
            Vold[i, 0] = V[i, 0]
            Vold[i, 1] = V[i, 1]
            
        update_eigenvalues(lambdas, Vold, tol)
        dt = min(0.5 * dx / lambdas.max(), tfinal - t)

        LaxFriedrich(V, Vold, lambdas, dt, dx, tol)

        t += dt
        
    return t



domain = np.array([0, 1])
T = 2.0
nstart = 256
nstep = 10

with open("RunningOn" + socket.gethostname(), "w") as f:
    for i in range(nstep):
        Nx = nstart * 2**i
        dx = (domain[1] - domain[0]) / Nx

        t = 0.
        V = np.empty((Nx, 2))
        Vold = np.empty_like(V)
        lambdas = np.empty(Nx)

        tol = np.finfo(np.float64).eps

        if i == 0:
            init_solution(V, domain)
            print("Warming up...", end=" ")
            update_to_time(t, tol, V, Vold, lambdas, dx, tol)
            print("Done")

        init_solution(V, domain)
        
        #print(f"Initial time t = {t}")
        print(f"Running with N = {Nx} :", end=" ")
        t_start = time.time()
        t = update_to_time(t, T, V, Vold, lambdas, dx, tol)
        t_end = time.time()
        print(f"{t_end - t_start}")
        #print(f"End of simulation, t = {t}")
        #print(f"Elapsed time: {t_end - t_start}")

        #print(f"mean(h) = {np.mean(V[:, 0])}")
        #np.savetxt("sol-cpu", V[:, 0])
        f.write(str(Nx) + "\t" + str(t_end - t_start) + "\n")
