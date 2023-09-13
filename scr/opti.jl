using Distances
using Optim
using LaTeXStrings

function loss(K::Matrix, ψ::Vector, q::Vector, ε::Real, Ψₘ::Real=0.2)
    @assert length(collect(axes(K)[2])) == length(ψ)
    @assert length(collect(axes(K)[1])) == length(q)
    sqeuclidean(K * ψ, q) + ε * norm(ψ .- Ψₘ)^2
end

defaultInit(ψ::Vector) = zero(ψ)


function optimizePSD(K::Matrix, q::Vector, ε::Real, method=LBFGS(), ψ₀=defaultInit(length(q)), loss=loss)
    Optim.optimize(ψ -> loss(K, ψ, q, ε), ψ₀; iterations=100000)
end

function plotOptiRes(rList, Ψₙᵤₘ, Ψₜᵣᵤₑ)
    plot(rList, Ψₙᵤₘ, label="Numerical Resolution", linewidth=3, dpi=300)#, yticks=0:0.4)
    plot!(rList, Ψₜᵣᵤₑ, label="Exact", linewidth=3, dpi=300)
    xlabel!("Particle Size")
    ylabel!("Probability density")
    title!("Particle Size Distribution")
end

function computeError(ψₙᵤₘ, ψₜᵣᵤₑ)
    sqeuclidean(ψₙᵤₘ, ψₜᵣᵤₑ)
end

function plotError(ψₙᵤₘ, ψₜᵣᵤₑ)

end
