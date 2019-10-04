# perf or animation ?
const INFO = false
const record_animation = false


using Printf
#using Makie
using Statistics
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





# initialization
const domain = Domain(0., 1.)
const Nx = 2^14


# loop in time
const T = 2.0
const dx = (domain.x_end - domain.x_start) / Nx
const tol = eps(Float64)

let t=0
    V = Array{Float64, 2}(undef, 2, Nx)
    InitSolution!(V, domain)
    
    Vold = Array{Float64, 2}(undef, 2, Nx)
    lambdas = Array{Float64, 1}(undef, Nx)

    function update_to_time(tframe::Float64, tol)
        while(t < tframe)
            @. Vold = V
            update_eigenvalues!(lambdas, Vold, tol)
            dt = min(0.5 * dx / maximum(lambdas), tframe - t)
            LaxFriedrich!(V, Vold, lambdas, dt, dx, tol)
            t += dt
            INFO && @printf("\t%.5f / %.5f\n", t, tframe)
        end
    end

    if record_animation
        nframe_per_second = 1000
        nframe = convert(Int64, ceil(nframe_per_second * T)) + 1

        scene = Scene(resolution = (1920, 1080))
        x = range(domain.x_start, stop=domain.x_end, length=Nx)

        plt = lines!(scene, x, view(V, 1, :), color=:blue, linewidth=2)[end]
        record(scene, "./height1d.mp4", range(t, stop=T, length=nframe), framerate=30) do tframe
            println(tframe, " / ", T)
            update_to_time(tframe, tol)
            plt[2] = view(V, 1, :)
        end
    else
        print("Warming up... ")
        update_to_time(eps(Float64), tol)
        println("Done")

        t = 0
        InitSolution!(V, domain)

        println("Initial time t = ", t)
        @time update_to_time(T, tol)
        println("End of simulation, t = ", t)
        
        println("mean(h) = ", mean(V[1, :]))
        writedlm("sol-cpu", V[1, :], "\n")
    end
end
