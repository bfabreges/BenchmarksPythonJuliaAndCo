using Printf
using DelimitedFiles




# the type for the domain
struct Domain
    x_start::Float64
    x_end::Float64
end


# Initial solution
function InitSolution!(V::Array{Float64, 2}, domain::Domain)
    nx = size(V, 2)

    hx = (domain.x_end - domain.x_start) / nx
    xc = 0.5  * (domain.x_start + domain.x_end) + nx/8 * hx
    
    for i=1:nx
        x = domain.x_start + (i-0.5) * hx    
        if abs(x - xc) < 0.2
            V[1, i] = 1
        else
            V[1, i] = 0
        end

        V[2, i] = 0
    end
end


# Model
function F!(f, V1, V2, tol)
    f[1] = V2
    if V1 < tol
        f[2] = 0
    else
        f[2] = V2^2 / V1 + 0.5 * 9.81 * V1^2
    end
end


# Scheme
function LaxFriedrich!(V::Array{Float64, 2}, Vold::Array{Float64, 2}, lambdas::Array{Float64, 1}, dt, dx, tol)
    nx = size(V, 2)
    Cx = dt / dx
    
    flux1 = Array{Float64}([0., 0.])
    flux2 = Array{Float64}([0., 0.])


    F!(flux1, Vold[1, 1], Vold[2, 1], tol)

    # left boundary
    V[2, 1] += Cx * (flux1[2] - lambdas[1] * Vold[2, 1])

    # interior edges
    for i=1:nx-1
        F!(flux2, Vold[1, i+1], Vold[2, i+1], tol)
        lambda = max(lambdas[i], lambdas[i+1])
        
        flux = 0.5 * Cx * ((flux2[1] + flux1[1]) - lambda * (Vold[1, i+1] - Vold[1, i]))
        V[1, i] -= flux
        V[1, i+1] += flux
        
        flux = 0.5 * Cx * ((flux2[2] + flux1[2]) - lambda * (Vold[2, i+1] - Vold[2, i]))
        V[2, i] -= flux
        V[2, i+1] += flux
        
        flux1, flux2 = flux2, flux1
    end

    # right boundary
    V[2, nx] -= Cx * (flux1[2] + lambdas[nx] * Vold[2, nx])
end

# Update the eigenvalues lambdas
function update_eigenvalues!(lambdas::Array{Float64, 1}, V::Array{Float64, 2}, tol::Float64)
    nx = size(V, 2)
    
    for i=1:nx
        if V[1, i] < tol
            lambdas[i] = 0.
        else
            lambdas[i] = abs(V[2, i] / V[1, i]) + sqrt(9.81 * V[1, i])
        end
    end
end



function update_to_time(V::Array{Float64, 2}, Vold::Array{Float64, 2}, lambdas::Array{Float64, 1},
                        t::Float64, tframe::Float64, dx::Float64, tol::Float64)
    while(t < tframe)
        @. Vold = V
        update_eigenvalues!(lambdas, Vold, tol)
        dt = min(0.5 * dx / maximum(lambdas), tframe - t)
        LaxFriedrich!(V, Vold, lambdas, dt, dx, tol)
        t += dt
    end
end




# initialization
const domain = Domain(0., 1.)

# loop in time
const T = 2.0
const tol = eps(Float64)

const nstart = 256
const nstep = 10

open("RunningOn" * gethostname(), "w") do file
    for istep=0:nstep-1
        Nx = nstart * 2^istep
        dx = (domain.x_end - domain.x_start) / Nx
    
        V = Array{Float64, 2}(undef, 2, Nx)        
        Vold = Array{Float64, 2}(undef, 2, Nx)
        lambdas = Array{Float64, 1}(undef, Nx)
        
        if istep == 0
            InitSolution!(V, domain)
            print("Warming up... ")
            update_to_time(V, Vold, lambdas, 0., eps(Float64), dx, tol)
            println("Done")
        end
        
        t = 0
        InitSolution!(V, domain)
        
        #println("Initial time t = ", t)
        print("Running with N = ", Nx, " : ")
        t = @elapsed update_to_time(V, Vold, lambdas, 0., T, dx, tol)
        println(t, "s")
        #println("End of simulation, t = ", t)
        
        #println("mean(h) = ", mean(V[1, :]))
        #writedlm("sol-cpu", V[1, :], "\n")
        write(file, string(Nx), "\t", string(t), "\n")
    end
end
