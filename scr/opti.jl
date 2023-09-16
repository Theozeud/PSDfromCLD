
using Distances
using Optim
using LaTeXStrings

# This is the loss function to compute 
function lossCLD(K::Matrix, ψ::Vector, q::Vector, ε::Real, Ψₘ::Real)
    @assert length(collect(axes(K)[2])) == length(ψ)
    @assert length(collect(axes(K)[1])) == length(q)
    sqeuclidean(K * ψ, q) + ε * norm(ψ .- Ψₘ)^2
end

defaultInit(ψ::Vector) = zero(ψ)

function optimizePSD(K::Matrix, q::Vector, ε::Real, ψₘ::Real=0, method=Nel(), ψ₀=defaultInit(length(q)), loss=loss)
    Optim.optimize(ψ -> lossCLD(K, ψ, q, ε, ψₘ), ψ₀; iterations=100000)
end

function plotOptiRes(rVector::Vector, ψₙᵤₘ::Vector, ψₜᵣᵤₑ::Vector)
    plot(rVector, ψₙᵤₘ, label="Numerical Resolution", linewidth=3, dpi=300)#, yticks=0:0.4)
    plot!(rVector, ψₜᵣᵤₑ, label="Exact", linewidth=3, dpi=300)
    xlabel!("Particle Size")
    ylabel!("Probability density")
    title!("Particle Size Distribution")
end

function plotOptiRes(rVector::Vector, ψₙᵤₘ::Vector, ψₜᵣᵤₑ::Vector, ψ₀::Vector)
    plot(rVector, ψₙᵤₘ, label="Numerical Resolution", linewidth=3, dpi=300)#, yticks=0:0.4)
    plot!(rVector, ψₜᵣᵤₑ, label="Exact", linewidth=3, dpi=300)
    plot!(rVector, ψ₀, label="Initial Distribution", linewidth=3, dpi=300)
    xlabel!("Particle Size")
    ylabel!("Probability density")
    title!("Particle Size Distribution")
end

function plotError(rVector, ψₙᵤₘ, ψₜᵣᵤₑ)
    error = [sqrt(((ψₙᵤₘ[i] - ψₜᵣᵤₑ[i]))^2) for i in 1:length(ψₙᵤₘ)]
    plot(rVector, error, linewidth=3, dpi=300, label=L"\Psi_{\text{num}} - \Psi_{\text{exact}}")
    xlabel!("Particle Size")
    ylabel!("Error")
    title!("Error in the Numerical Resolution")
end
