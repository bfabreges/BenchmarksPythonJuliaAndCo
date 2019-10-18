#!/usr/bin/python3

import numpy as np
import socket
from pathlib import Path
import matplotlib.pyplot as plt

def parsit(D, name, l): 
    D[name] = float(l.replace("\n", ""))

# directories to explore
tests = {
    "C++" : "ob-",
    "Ju" : "or-",
    "Python" : "og--",
    "Python-Numba" : "og-",
    "C++-OpenMP" : "xb-",
    "Ju-Threads" : "xr-",
    "Python-Numba-Threads" : "xg-",
    "C++-CUDA" : "b",
    "Ju-CUDA" : "r",
    "Python-Numba-CUDA" : "g"
}

ref = "C++"

# build a dict  n -> computing time for reference
timings = np.loadtxt("../" + ref + "/RunningOn" + socket.gethostname())
timings_ref = dict(zip(np.int64(timings[:, 0]), timings[:, 1]))

legend = []
fig = plt.figure(figsize=[15, 10], dpi=300)
ax = fig.subplots()
ax.set_xscale('log', basex=2)
ax.set_yscale('log')
plt.grid(which='both', axis='y', alpha=0.5)
for testname, plot_prop in tests.items():
    filename =  "../" + testname + "/RunningOn" + socket.gethostname()
    try:
        timings = np.loadtxt(filename)
        n_test = np.int64(timings[:, 0])
        t_test = timings[:, 1]
        for i in range(n_test.size):
            if n_test[i] in timings_ref:
                t_test[i] /= timings_ref[n_test[i]]
            else:
                print(f"Error, the reference ({ref} test) has not been run with {n_test[i]} points")
                exit(1)

        plt.plot(n_test, t_test, plot_prop)
        legend.append(testname)

    except OSError:
        print(f"{filename} does not exists ! Did you run this test ?")

plt.xlabel("Number of discretization points")
plt.ylabel(f"Time compared with {ref}")
plt.title("Time comparison for 1D shallow water equations")
plt.legend(legend)
plt.savefig("results.png")
