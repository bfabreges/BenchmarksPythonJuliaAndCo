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
def model(V1, V2, tol):
    if V1 < tol:
        return 0.
    else:
        return V2**2 / V1 + 0.5 * 9.81 * V1**2


@numba.njit(parallel=True)
def LaxFriedrich(V, Vold, lambdas, dt, dx, tol):
    nx = V.shape[0]
    Cx = dt / dx

    f_m = model(Vold[0, 0], Vold[0, 1], tol)
    f_p = model(Vold[1, 0], Vold[1, 1], tol)
    ml = max(lambdas[0], lambdas[1])

    V[0, 0] -= 0.5 * Cx * ((Vold[1, 1] + Vold[0, 1]) - ml * (Vold[1, 0] - Vold[0, 0]));
    V[0, 1] += Cx * (f_m - lambdas[0] * Vold[0, 1]) - 0.5 * Cx * ((f_p + f_m) - ml * (Vold[1, 1] - Vold[0, 1]));
    
    for i in numba.prange(1, nx-1):
        f_hu_m = model(Vold[i-1, 0], Vold[i-1, 1], tol);
        f_hu_p = model(Vold[i+1, 0], Vold[i+1, 1], tol);
        mlm = max(lambdas[i-1], lambdas[i]);
        mlp = max(lambdas[i], lambdas[i+1]);

        V[i, 0] += 0.5 * Cx * ((Vold[i-1, 1] - Vold[i+1, 1]) - mlm * (Vold[i, 0] - Vold[i-1, 0]) + mlp * (Vold[i+1, 0] - Vold[i, 0]));
        V[i, 1] += 0.5 * Cx * ((f_hu_m - f_hu_p) - mlm * (Vold[i, 1] - Vold[i-1, 1]) + mlp * (Vold[i+1, 1] - Vold[i, 1]));

    f_m = model(Vold[Nx-2, 0], Vold[Nx-2, 1], tol);
    f_p = model(Vold[Nx-1, 0], Vold[Nx-1, 1], tol);
    ml = max(lambdas[Nx-2], lambdas[Nx-1]);

    V[Nx-1, 0] += 0.5 * Cx * ((Vold[Nx-1, 1] + Vold[Nx-2, 1]) - ml * (Vold[Nx-1, 0] - Vold[Nx-2, 0]));
    V[Nx-1, 1] += 0.5 * Cx * ((f_p + f_m) - ml * (Vold[Nx-1, 1] - Vold[Nx-2, 1])) - Cx * (f_p + lambdas[Nx-1] * Vold[Nx-1, 1]);
        

@numba.njit(parallel=True)
def update_eigenvalues(lambdas, V, tol):
    nx = lambdas.size
    for i in numba.prange(nx):
        if V[i, 0] < tol:
            lambdas[i] = 0.
        else:
            lambdas[i] = np.abs(V[i, 1] / V[i, 0]) + np.sqrt(9.81 * V[i, 0])


@numba.njit(parallel=True)
def update_to_time(t, tfinal, V, Vold, lambdas, dx, tol):
    while t < tfinal:
        for i in numba.prange(V.shape[0]):
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
