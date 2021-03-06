### Description:

All programs perform a  _hand coded_ Gaussian elimination with partial pivoting, except those
in 
C++Lib/, PyScipy/ and JuLib/ which call the Lapack and blas libraries  to perform
a LU factorization; in any case, the cost is similar in term of operations counts. 

### Motivation:

The  Gaussian elimination with partial pivoting, even if this is something one
should not code in the real life (except for pedagogical purpose) seems to be
a good prototype of numerical algorithm an user would like to code (as it uses
about n^3 floating point operations and array manipulations).

### Authors:

Thierry Dumont   tdumont@math.univ-lyon1.fr

Benoît Fabrèges  fabreges@math.univ-lyon1.fr

Roland Denis     denis@math.univ-lyon1.fr

### The directories contain:

- **C++**:      pure C++, sequential.

- **C++Lib**:   C++ and lapack + openblas.

- **Ju**:       pure Julia

- **JuLib**:    Julia +  lapack + openblas.


- **Py**:       pure Python

- **PyVec**:    pure Python, vectorized.

- **PyScipy**:  Python + Scipy.

- **Pythran**:  Pythran.

- **PythranVec**: Pythran vectorized

- **Numba**:    Python + Numba

- **Results**:    graphical material and an example of result.

### Note:

* Concerning C++Lib/,  JuLib/ and PyScipy/ all of them call lapack  material and
blas which probably will be openblas. For large matrix sizes, with openblas, the
called routines would possibly be multithreaded at least in some cases (see
PLOT/Benchmark/kepler.pdf, for which C++Lib is multithreaded).

* Concerning the random number generator: we use a very poor
generator -and not a genarator provided by any language dependent
library- only because we want to be sure (and we actually checked
this) than the "random" sequences generated are identical in all
languages. Actually, we are not interested in "good" random sequences. 

### Running the benchmarks

cd succesively in C++, Ju, Py, Numba, Pythran, PyVec, C++Lib, JuLib,
PyScipy PythranVec; then look at the documentation.

#### Results:

Running the codes  will create a file named:  RunningOn"your
hostname".

### Ploting results:

cd PLOT/

Read README.md file.


### Results:
Have a look at the  [this page in the Wiki](https://github.com/Thierry-Dumont/BenchmarksPythonJuliaAndCo/wiki/3-The-Gaussian-Benchmark).
