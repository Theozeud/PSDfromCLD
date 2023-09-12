
# Known distribution for testing the compution of PSD
abstract type RandomShape end

struct Normal <: RandomShape 
    m::Real
    σ::Real
end

law(n::Normal,r::Real) = 1/sqrt(2*n.σ^2)*exp(-(r-n.m)^2/(2*n.σ^2))
tir(n::Normal, m::Int = 1) = randn(m)*n.σ.+n.m

struct Uniform <: RandomShape 
    sup::Real
    inf::Real
end

law(u::Uniform, r::Real) = 1/(u.sup-u.inf)
tir(u::Uniform, n::Int = 1) = rand(n)*(u.sup-u.inf) .+ u.inf

