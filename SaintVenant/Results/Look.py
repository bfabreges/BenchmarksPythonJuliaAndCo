#!/usr/bin/python3

import numpy as np
import socket
from pathlib import Path
import matplotlib.pyplot as plt

def parsit(D, name, l): 
    D[name] = float(l.replace("\n", ""))

# directories to explore
files = [
    "C++",
    "Ju",
    "Python",
    "Python-Numba",
    "C++-CUDA",
    "Ju-CUDA",
    "Python-Numba-CUDA"
]

ref = "C++"


# build a dict  n -> computing time for reference
timings = np.loadtxt("../" + ref + "/RunningOn" + socket.gethostname())
timings_ref = dict(zip(np.int64(timings[:, 0]), timings[:, 1]))

legend = []
for f in files:
    filename =  "../" + f + "/RunningOn" + socket.gethostname()
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

        plt.loglog(n_test, t_test)
        legend.append(f)

    except OSError:
        print(f"{filename} does not exists ! Did you run this test ?")

plt.legend(legend)
plt.savefig("results.png")
