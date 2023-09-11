
# Known distribution for testing the compution of PSD

normal(n::Int, m::Real, σ::Real) = randn(n)*σ+m

uniform(n::Int, a::inf, b::sup) = rand(n)*(b-a)+a