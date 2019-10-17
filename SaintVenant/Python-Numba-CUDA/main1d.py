import numpy as np
import math
from numba import cuda, njit, float64, int32
import time
import socket



def init_solution(V, domain):
    nx = V.shape[0]
    hx = (domain[1] - domain[0]) / nx
    xc = 0.5 * (domain[0] + domain[1]) + nx / 8 * hx

    V[:, 0] = np.abs(domain[0] + (np.arange(nx) + 0.5) * hx - xc) < 0.2
    V[:, 1] = 0




@cuda.jit(argtypes=(float64[:], float64[:]))
def run_max(d_in, d_out):
    N = d_in.size
    
    # assuming that nthreads is >= 64
    smax = cuda.shared.array(shape=0, dtype=float64)
    
    gid = cuda.blockIdx.x * cuda.blockDim.x * 2 + cuda.threadIdx.x
    tid = cuda.threadIdx.x

    if gid < N:
        mm = d_in[gid]
        if gid + cuda.blockDim.x < N:
            mm = max(mm, d_in[gid + cuda.blockDim.x])
    else:
        mm = -1.0

    smax[tid] = mm
    cuda.syncthreads()

    if cuda.blockDim.x >= 512 and tid < 256:
        smax[tid] = mm = max(mm, smax[tid + 256])
    cuda.syncthreads()
    
    if cuda.blockDim.x >= 256 and tid < 128:
        smax[tid] = mm = max(mm, smax[tid + 128])
    cuda.syncthreads()

    if cuda.blockDim.x >= 128 and tid < 64:
        smax[tid] = mm = max(mm, smax[tid + 64])
    cuda.syncthreads()

    if tid < 32:
        if cuda.blockDim.x >= 64:
            smax[tid] = mm = max(mm, smax[tid + 32])

        for offset in [16, 8, 4, 2, 1]:
            mm = max(mm, cuda.shfl_down_sync((1 << 2*offset) - 1, mm, offset))

    if tid == 0:
        d_out[cuda.blockIdx.x] = mm
        
        
    
@cuda.jit(argtypes=(float64, float64, float64), device=True)
def model(V1, V2, tol):
    if V1 < tol:
        return 0.
    else:
        return V2**2 / V1 + 0.5 * 9.81 * V1**2


@cuda.jit(argtypes=(float64[:, :], float64[:, :], float64[:], float64, float64, float64, int32))
def LaxFriedrich(d_V, d_Vold, d_lambdas, dt, dx, tol, smem):
    sdata = cuda.shared.array(shape=0, dtype=float64)

    gid = cuda.grid(1)
    lid = cuda.threadIdx.x + 1

    nx = d_V.shape[0]
    Cx = dt / dx

    if gid < nx:
        sdata[lid] = d_Vold[gid, 0]
        sdata[smem + lid] = d_Vold[gid, 1]
        sdata[2*smem + lid] = d_lambdas[gid]

        if lid == 1 and cuda.blockIdx.x > 0:
            sdata[0] = d_Vold[gid - 1, 0]
            sdata[smem] = d_Vold[gid - 1, 1]
            sdata[2*smem] = d_lambdas[gid - 1]

        if lid == cuda.blockDim.x and cuda.blockIdx.x < cuda.gridDim.x - 1:
            sdata[lid + 1] = d_Vold[gid + 1, 0]
            sdata[smem + lid + 1] = d_Vold[gid + 1, 1]
            sdata[2*smem + lid + 1] = d_lambdas[gid + 1]

    cuda.syncthreads()

    if gid > 0 and gid < nx - 1:
        lid_s = smem + lid
        lid_ss = 2 * smem + lid
        f_hu_m = model(sdata[lid-1], sdata[lid_s-1], tol)
        f_hu_p = model(sdata[lid+1], sdata[lid_s+1], tol)
        mlm = max(sdata[lid_ss-1], sdata[lid_ss])
        mlp = max(sdata[lid_ss], sdata[lid_ss+1])

        d_V[gid, 0] += 0.5 * Cx * ((sdata[lid_s-1] - sdata[lid_s+1]) - mlm * (sdata[lid] - sdata[lid-1]) + mlp * (sdata[lid+1] - sdata[lid]))
        d_V[gid, 1] += 0.5 * Cx * ((f_hu_m - f_hu_p) - mlm * (sdata[lid_s] - sdata[lid_s-1]) + mlp * (sdata[lid_s+1] - sdata[lid_s]))
    elif gid == 0:
        f_hu_m = model(sdata[1], sdata[smem+1], tol)
        f_hu_p = model(sdata[2], sdata[smem+2], tol)
        ml = max(sdata[2*smem+1], sdata[2*smem+2])

        d_V[0, 0] -= 0.5 * Cx * ((sdata[smem+2] + sdata[smem+1]) - ml * (sdata[2] - sdata[1]))
        d_V[0, 1] += Cx * (f_hu_m - sdata[2*smem+1] * sdata[smem+1]) - 0.5 * Cx * ((f_hu_p + f_hu_m) - ml * (sdata[smem+2] - sdata[smem+1]))
    elif gid == nx - 1:
        lid_s = smem + lid
        lid_ss = 2 * smem + lid
        f_hu_m = model(sdata[lid-1], sdata[lid_s-1], tol)
        f_hu_p = model(sdata[lid], sdata[lid_s], tol)
        ml = max(sdata[lid_ss-1], sdata[lid_ss])

        d_V[nx-1, 0] += 0.5 * Cx * ((sdata[lid_s] + sdata[lid_s-1]) - ml * (sdata[lid] - sdata[lid-1]))
        d_V[nx-1, 1] += 0.5 * Cx * ((f_hu_p + f_hu_m) - ml * (sdata[lid_s] - sdata[lid_s-1])) - Cx * (f_hu_p + sdata[lid_ss] * sdata[lid_s]);
        

        
@cuda.jit(argtypes=(float64[:], float64[:, :], float64))
def update_eigenvalues(d_lambdas, d_V, tol):
    nx = d_lambdas.size
    i = cuda.grid(1)
    if i < nx:
        if d_V[i, 0] < tol:
            d_lambdas[i] = 0.
        else:
            d_lambdas[i] = abs(d_V[i, 1] / d_V[i, 0]) + math.sqrt(9.81 * d_V[i, 0])
            

@cuda.jit(argtypes=(float64[:, :], float64[:, :]))
def copy_device_to_device(d_a, d_b):
    nx = d_a.size
    i = cuda.grid(1)
    if i < nx:
        d_a[i, 0] = d_b[i, 0]
        d_a[i, 1] = d_b[i, 1]

            

def update_to_time(t, tfinal, d_V, d_Vold, d_lambdas, d_lmax, lmax, dx, tol, nblocks, nblocks_max, nthreads):
    while t < tfinal:
        #t_start = time.time()
        copy_device_to_device[nblocks, nthreads, 0, 0](d_Vold, d_V)
        #t_end = time.time()
        #print(f"Copy: {t_end - t_start}")

        #t_start = time.time()
        update_eigenvalues[nblocks, nthreads, 0, 0](d_lambdas, d_Vold, tol)
        #t_end = time.time()
        #print(f"Eigenvalues: {t_end - t_start}")

        #t_start = time.time()
        run_max[nblocks_max, nthreads, 0, smem_max](d_lambdas, d_lmax)
        d_lmax.copy_to_host(lmax)
        Cmax = np.amax(lmax)
        #t_end = time.time()
        #print(f"Max: {t_end - t_start}")

        dt = min(0.5 * dx / Cmax, tfinal - t)
        
        #t_start = time.time()
        LaxFriedrich[nblocks, nthreads, 0, smem_scheme](d_V, d_Vold, d_lambdas, dt, dx, tol, n_smem_elem)
        #t_end = time.time()
        #print(f"Lax-Friedrich: {t_end - t_start}")

        t += dt

    return t


cuda.detect()
print()

domain = np.array([0, 1])
T = 2.0

nstart = 256
nstep = 10

with open("RunningOn" + socket.gethostname(), "w") as f:
    for i in range(nstep):
        Nx = nstart * 2**i
        dx = (domain[1] - domain[0]) / Nx

        nthreads = 256

        nblocks = (Nx + (nthreads - 1)) // nthreads
        nblocks_max = (nblocks + 1) // 2

        smem_max = nthreads * np.dtype(np.float64).itemsize
        n_smem_elem = nthreads + 2
        smem_scheme = 3 * n_smem_elem * np.dtype(np.float64).itemsize

        t = 0.
        V = np.empty((Nx, 2), dtype=np.float64)

        d_V = cuda.to_device(V)
        d_Vold = cuda.device_array_like(d_V)
        d_lambdas = cuda.device_array(Nx, dtype=np.float64)

        #lmax = np.empty(1, dtype=np.float64)
        #d_lmax = cuda.to_device(lmax)
        #run_max = cuda.Reduce(lambda a, b: max(a, b))  # too slow
        lmax = np.empty(nblocks_max, dtype=np.float64)
        d_lmax = cuda.to_device(lmax)

        tol = np.finfo(np.float64).eps

        if i == 0:
            init_solution(V, domain)
            d_V.copy_to_device(V)
            print("Warming up...", end=" ")
            update_to_time(t, tol, d_V, d_Vold, d_lambdas, d_lmax, lmax, dx, tol, nblocks, nblocks_max, nthreads)
            print("Done")


        init_solution(V, domain)
        
        #print(f"Initial time t = {t}")
        print(f"Running with N = {Nx} :", end=" ")
        t_start = time.time()
        #cuda.profile_start()
        d_V.copy_to_device(V)
        t = update_to_time(t, T, d_V, d_Vold, d_lambdas, d_lmax, lmax, dx, tol, nblocks, nblocks_max, nthreads)
        d_V.copy_to_host(V)
        #cuda.profile_stop()
        t_end = time.time()
        print(f"{t_end - t_start}")
        #print(f"End of simulation, t = {t}")
        #print(f"Elapsed time: {t_end - t_start}")

        #d_V.copy_to_host(V)
        #print(f"mean(h) = {np.mean(V[:, 0])}")
        #np.savetxt("sol-cpu", V[:, 0])
        f.write(str(Nx) + "\t" + str(t_end - t_start) + "\n")
