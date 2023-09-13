using Plots
using LaTeXStrings
include("../scr/utils.jl")

# Known distribution for testing the compution of PSD
abstract type RandomShape end

function plotLaw(rs::RandomShape, ntirage::Int; title::Bool=false)
    xmin, xmax = interval(rs)
    x = range(xmin, xmax, ntirage)
    y = apply(x -> law(rs, x), x)
    plot(x, y, label=shortTitle(rs))
    xlabel!(L"x")
    ylabel!(L"pdf(x)")
    #if title
    #title!(title(rs))
    #end
end

struct Normal <: RandomShape
    m::Real
    σ::Real
end

interval(n::Normal) = (n.m - 3 * n.σ, n.m + 3 * n.σ)
law(n::Normal, r::Real) = 1 / ((n.σ) * sqrt(2 * π)) * exp(-(r - n.m)^2 / (2 * (n.σ)^2))
tir(n::Normal, m::Int=1) = randn(m) * n.σ .+ n.m
title(n::Normal) = string("Normal law with mean ", n.m, " and standard deviation ", n.σ)
shortTitle(n::Normal) = string(L"\mathcal{N}(", n.m, ",", n.σ, L")")

struct Uniform <: RandomShape
    inf::Real
    sup::Real
end

interval(u::Uniform) = (u.inf, u.sup)
law(u::Uniform, r::Real) = u.inf <= r && u.sup >= r ? 1 / (u.sup - u.inf) : 0.0
tir(u::Uniform, n::Int=1) = rand(n) * (u.sup - u.inf) .+ u.inf
title(u::Uniform) = string("Uniform law between ", u.inf, " and ", u.sup)
shortTitle(u::Uniform) = string(L"\mathcal{U}(", u.inf, ",", u.sup, L")")
