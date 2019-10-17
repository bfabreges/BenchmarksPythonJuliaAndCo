#include <iostream>
#include <cmath>
#include <functional>
#include <chrono>
#include <fstream>

#include <unistd.h>
#include <limits.h>

#include <Eigen/Dense>

#include "main1d-kernels.hpp"

struct Domain {
    constexpr Domain(double start, double end) : x_start(start), x_end(end) {}
    double x_start;
    double x_end;
};


template <typename Derived>
void init_solution(Eigen::ArrayBase<Derived>& V1, Eigen::ArrayBase<Derived>& V2, const Domain& domain) {
    const std::size_t nx = V1.size();
    const double hx = (domain.x_end - domain.x_start) / nx;
    const double xc = 0.5 * (domain.x_start + domain.x_end) + (nx / 8.0) * hx;

    for(std::size_t i=0; i < nx; ++i) {
        double x = domain.x_start + (i - 0.5) * hx;
        V1(i) = std::abs(x - xc) < 0.2 ? 1. : 0.;
        V2(i) = 0.;
    }           
}


template<typename ArrayType>
double reduce(double* lambdas, double* lmb_max, std::size_t size, ArrayType& lmax, std::size_t num_blocks, std::size_t num_threads, std::size_t smem_size) 
{
    // compute the maximum for each block
    reduce_max<<<num_blocks, num_threads, smem_size>>>(lambdas, lmb_max, size);

    // finishing on cpu. What if NUM_BLOCK >> NUM_THREADS ? Re-run a kernel ?
    cudaMemcpy(lmax.data(), lmb_max, lmax.size() * sizeof(double), cudaMemcpyDeviceToHost);
    return lmax.maxCoeff();
}



template<typename ArrayType>
double update_to_time(double t, double tframe, double dx,
                      double* V1, double* V2,
                      double* Vold1, double* Vold2,
                      double* lambdas,
                      double* lmb_max, ArrayType& lmax,
                      double tol, std::size_t Nx,
                      std::size_t num_blocks, std::size_t num_threads,
                      std::size_t nblocks_max, std::size_t smem_max)
{
    std::size_t n = 0;
    while(t < tframe) {
        update_eigenvalues<<<num_blocks, num_threads>>>(lambdas, V1, V2, tol, Nx);
        double Cmax = reduce(lambdas, lmb_max, Nx, lmax, nblocks_max, num_threads, smem_max);
        double dt = std::min(0.5 * dx / Cmax, tframe - t);
        
        cudaMemcpy(Vold1, V1, Nx * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(Vold2, V2, Nx * sizeof(double), cudaMemcpyDeviceToDevice);
        
        scheme_LaxFriedrich<<<num_blocks, num_threads, 3*(num_threads + 2) * sizeof(double)>>>(V1, V2, Vold1, Vold2, lambdas, dt, dx, tol, Nx);
        
        t += dt;
        ++n;
    }

    //std::cout << "Number of iterations:" << n << std::endl;
    return t;
}
                                       


int main() {
    constexpr Domain domain(0., 1.);
    const double T = 2.0;
    const double tol = 1e-15;
    const std::size_t nstart = 256;
    const std::size_t nstep = 10;
    
    char hostnameC[HOST_NAME_MAX];
    gethostname(hostnameC, HOST_NAME_MAX);
    std::string hostname = hostnameC;

    std::ofstream timings("RunningOn" + hostname);

    const std::size_t num_gpu_threads = 256;
    const std::size_t smem_max = num_gpu_threads <= 32 ? 64 * sizeof(double) : num_gpu_threads * sizeof(double);
    
    std::size_t Nx = nstart;
    for(std::size_t i=0; i < nstep; ++i, Nx *= 2) {
        const double dx = (domain.x_end - domain.x_start) / Nx;

        std::modulus<std::size_t> mod;
        const std::size_t num_gpu_blocks = mod(Nx, num_gpu_threads) == 0 ? Nx / num_gpu_threads : Nx / num_gpu_threads + 1;
        const std::size_t nblocks_max = mod(num_gpu_blocks, 2) == 0 ? num_gpu_blocks / 2 : num_gpu_blocks / 2 + 1;
    
        //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
        Eigen::Array<double, 1, Eigen::Dynamic> lmax(1, nblocks_max);
        Eigen::Array<double, 1, Eigen::Dynamic> V1(1, Nx);
        Eigen::Array<double, 1, Eigen::Dynamic> V2(1, Nx);
        init_solution(V1, V2, domain);
    
        double *d_V1, *d_V2;
        cudaMalloc((void**)&d_V1, Nx * sizeof(double));
        cudaMalloc((void**)&d_V2, Nx * sizeof(double));
        
        double *d_Vold1, *d_Vold2;
        cudaMalloc((void**)&d_Vold1, Nx * sizeof(double));
        cudaMalloc((void**)&d_Vold2, Nx * sizeof(double));

        double* d_lambdas;
        cudaMalloc((void**)&d_lambdas, Nx * sizeof(double));

        double* d_lmb_max;
        cudaMalloc((void**)&d_lmb_max, nblocks_max * sizeof(double));

        double t = 0;
        std::cout << "Running with N = " << Nx << " : ";
        //std::cout << "Initial time t = " << t << std::endl;
    
        auto time_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_V1, V1.data(), Nx * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V2, V2.data(), Nx * sizeof(double), cudaMemcpyHostToDevice);

        t = update_to_time(t, T, dx,
                           d_V1, d_V2,
                           d_Vold1, d_Vold2,
                           d_lambdas,
                           d_lmb_max, lmax,
                           tol, Nx,
                           num_gpu_blocks, num_gpu_threads, nblocks_max, smem_max);

        cudaMemcpy(V1.data(), d_V1, Nx * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(V2.data(), d_V2, Nx * sizeof(double), cudaMemcpyDeviceToHost);
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = time_end - time_start;

        //std::cout << "Elapsed time : " << elapsed_seconds.count() << "s" << std::endl;
        //std::cout << "End of simulation, t = " << t << std::endl;


        // copy back the result and compute the mean value
        //cudaMemcpy(V1.data(), d_V1, Nx * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(V2.data(), d_V2, Nx * sizeof(double), cudaMemcpyDeviceToHost);
        //std::cout << V1.mean() << std::endl;

        // write solution to file
        //const Eigen::IOFormat to_file(Eigen::FullPrecision, Eigen::DontAlignCols, "\n", "\n");
        //std::ofstream file("sol-gpu");
        //file << V1.format(to_file);

        std::cout << elapsed_seconds.count() << "s" << std::endl;
        timings << Nx << "\t" << elapsed_seconds.count() << std::endl;
        
        // freeing arrays on device
        cudaFree(d_V1); cudaFree(d_V2);
        cudaFree(d_Vold1); cudaFree(d_Vold2);
        cudaFree(d_lambdas);
        cudaFree(d_lmb_max);
    }

    timings.close();
    
    return 0;
}
